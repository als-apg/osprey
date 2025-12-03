=============
CLI Reference
=============

**What you'll learn:** Complete reference for all Osprey Framework CLI commands

.. dropdown:: 📚 What You'll Learn
   :color: primary
   :icon: book

   **Key Concepts:**

   - Using ``osprey`` CLI for all framework operations
   - Creating projects with ``osprey init``
   - Generating capabilities from MCP servers with ``osprey generate`` (prototype)
   - Managing deployments with ``osprey deploy``
   - Running interactive sessions with ``osprey chat``
   - Managing configuration with ``osprey config``

   **Prerequisites:** Framework installed (``pip install osprey-framework``)

   **Time Investment:** 10 minutes for quick reference

Overview
========

The Osprey Framework provides a unified CLI for all framework operations. All commands are accessed through the ``osprey`` command with subcommands for specific operations.


**Quick Reference:**

.. code-block:: bash

   osprey                    # Launch interactive menu (NEW in v0.7.7)
   osprey --version          # Show framework version
   osprey --help             # Show available commands
   osprey init PROJECT       # Create new project
   osprey generate COMMAND   # Generate components (MCP capabilities, servers)
   osprey deploy COMMAND     # Manage services
   osprey chat               # Start interactive chat
   osprey config             # Manage configuration

Interactive Mode
================

The framework provides an interactive terminal UI (TUI) that automatically launches when you run ``osprey`` without any arguments:

.. code-block:: bash

   osprey

The TUI is **completely optional** - all existing direct commands continue to work exactly as before. Use whichever approach fits your workflow:

- **Interactive mode**: Great for exploration, learning, and infrequent tasks
- **Direct commands**: Perfect for direct control without visual overhead, for experienced users

Context-Aware Menus
-------------------

The interactive menu automatically detects your context and adapts accordingly:

**Outside a Project Directory:**

When not in a project directory, the menu offers:

- **Select nearby project** - If framework projects are detected in subdirectories, you can select one to navigate to
- **Create new project (interactive)** - Guided project creation with template, provider, and model selection
- **Show init command syntax** - View direct CLI commands for scripting
- **Show main help** - Display all available commands
- **Exit** - Close the interactive menu

**Inside a Project Directory:**

When inside a project directory (detected by presence of ``config.yml``), the menu shows:

- **chat** - Start CLI conversation interface
- **deploy** - Manage containerized services (with subcommands for up, down, status, etc.)
- **health** - Run comprehensive system health check
- **config** - Display current project configuration
- **init** - Create a new project (in a different location)
- **help** - Show all available commands
- **exit** - Close the interactive menu

The menu displays your current project name and configuration (provider/model) for easy reference.

Interactive Project Creation
----------------------------

The interactive project creation flow guides you through all the necessary steps:

1. **Project Name**: Enter a name for your project
2. **Template Selection**: Choose from available templates with descriptions:

   - ``minimal`` - Basic skeleton for custom development
   - ``hello_world_weather`` - Simple weather agent example
   - ``control_assistant`` - Production control system integration template

3. **Provider Selection**: Select your AI provider (Cborg, OpenAI, Anthropic, Google, Ollama, etc.)
4. **Model Selection**: Choose from provider-specific models
5. **API Key Setup**:

   - Automatically detects API keys from your shell environment
   - Prompts for secure input if keys are not detected
   - Generates ``.env`` file with detected or entered keys

The interactive flow is equivalent to using ``framework init`` with appropriate flags, but with helpful guidance and validation at each step.

Disabling Interactive Mode
--------------------------

If you prefer to only use direct commands, you can bypass the interactive menu by:

- Running specific commands directly: ``framework chat``, ``framework deploy up``, etc.
- Using ``framework --help`` to see available commands
- The menu never interrupts existing scripts or automation

Global Options
==============

These options work with all ``osprey`` commands.

``--project`` / ``-p``
----------------------

The ``--project`` flag allows you to specify the project directory for commands that operate on existing projects (``chat``, ``deploy``, ``health``, ``config``), enabling multi-project workflows and CI/CD automation from any directory.

.. code-block:: bash

   osprey COMMAND --project /path/to/project

**Project Resolution Priority:**

When determining which project to use, the framework checks in this order:

1. **``--project`` CLI flag** (highest priority)
2. **``FRAMEWORK_PROJECT`` environment variable**
3. **Current working directory** (default)

**Examples:**

.. code-block:: bash

   # Work with specific project from anywhere
   osprey chat --project ~/projects/weather-agent
   osprey deploy status --project ~/projects/turbine-monitor

   # Use environment variable for a session
   export OSPREY_PROJECT=~/projects/my-agent
   osprey chat              # Uses ~/projects/my-agent
   osprey deploy status     # Uses ~/projects/my-agent

   # CLI flag overrides environment variable
   export OSPREY_PROJECT=~/projects/agent1
   osprey chat --project ~/projects/agent2  # Uses agent2, not agent1

**Use Cases:**

- **Multi-project development**: Switch between projects without changing directories
- **CI/CD pipelines**: Deploy or test specific projects from central scripts
- **Automation**: Run health checks across multiple projects
- **Parallel workflows**: Work with multiple projects simultaneously

**Commands supporting ``--project``:**

- ``osprey chat --project PATH``
- ``osprey deploy COMMAND --project PATH``
- ``osprey health --project PATH``
- ``osprey config --project PATH``

**Note**: The ``osprey init`` command does not use ``--project`` because it creates a new project. Use ``--output-dir`` instead to specify where the new project should be created.

``--version``
-------------

Show framework version and exit.

.. code-block:: bash

   osprey --version

Output:

.. code-block:: text

   Osprey Framework version 0.7.0

``--help``
----------

Show help message for any command.

.. code-block:: bash

   osprey --help
   osprey init --help
   osprey deploy --help
   osprey chat --help
   osprey config --help

CLI Customization
=================

You can customize the CLI appearance through the ``cli`` section in your project's ``config.yml``.

**Example - Custom colors:**

.. code-block:: yaml

   cli:
     theme: "custom"
     custom_theme:
       primary: "#4A90E2"      # Brand blue
       success: "#7ED321"      # Success green
       accent: "#F5A623"       # Accent orange
       command: "#9013FE"      # Command purple
       path: "#50E3C2"         # Path teal
       info: "#BD10E0"         # Info magenta

**Example - Custom banner:**

.. code-block:: yaml

   cli:
     theme: "default"
     banner: |
       ╔════════════════════════════════════════╗
       ║      MY PROJECT NAME                   ║
       ╚════════════════════════════════════════╝

All menu items, prompts, and messages will use your custom colors and banner. The default theme is used if no ``cli`` section is present.

Commands
========

osprey init
==============

Create a new project from a template.

Syntax
------

.. code-block:: bash

   osprey init [OPTIONS] PROJECT_NAME

Arguments
---------

``PROJECT_NAME``
   Name of the project directory to create. Will be created in the current directory.

Options
-------

``--template <name>``
   Template to use for project initialization. Available templates:

   - ``minimal`` - Basic skeleton for custom development
   - ``hello_world_weather`` - Simple weather agent (recommended for learning)
   - ``control_assistant`` - Production control system integration template

   Default: ``minimal``

``--registry-style <style>``
   Registry implementation style:

   - ``compact`` - Use helper functions (5-10 lines, recommended)
   - ``explicit`` - Full registry implementation (verbose, for learning)

   Default: ``compact``

Examples
--------

**Create minimal project:**

.. code-block:: bash

   osprey init my-agent

**Create from hello_world_weather template:**

.. code-block:: bash

   osprey init weather-demo --template hello_world_weather

**Create with explicit registry style:**

.. code-block:: bash

   osprey init my-agent --template minimal --registry-style explicit

**Create advanced agent:**

.. code-block:: bash

   osprey init my-assistant --template control_assistant

Generated Structure
-------------------

The ``osprey init`` command creates a complete, self-contained project:

.. code-block:: text

   my-agent/
   ├── src/
   │   └── my_agent/           # Application code
   │       ├── __init__.py
   │       ├── registry.py     # Component registration
   │       ├── context_classes.py
   │       └── capabilities/   # Agent capabilities
   ├── services/               # Container services
   │   ├── jupyter/           # Development environment
   │   ├── open-webui/        # Web interface
   │   └── pipelines/         # Processing pipeline
   ├── config.yml             # Complete configuration
   ├── .env.example           # Environment template
   └── README.md              # Project documentation

osprey generate
===============

.. admonition:: Prototype Feature
   :class: warning

   This is a **prototype feature** under active development. The API and generated code structure may change in future releases.

Generate Osprey components from various sources, including MCP (Model Context Protocol) servers.

Subcommands
-----------

``osprey generate capability``
   Generate Osprey capability from MCP server

``osprey generate mcp-server``
   Generate demo MCP server for testing

osprey generate capability
--------------------------

Generate a complete Osprey capability from an MCP server with automatic ReAct agent integration, classifier/orchestrator guides, and context classes.

**Syntax:**

.. code-block:: bash

   osprey generate capability --from-mcp <URL_OR_SIMULATED> --name <NAME> [OPTIONS]

**Required Arguments:**

``--from-mcp <url>``
   MCP server URL (e.g., ``http://localhost:3001``) or ``simulated`` for demo mode with weather tools

``--name, -n <name>``
   Name for the generated capability (e.g., ``slack_mcp``, ``weather_mcp``)

**Optional Arguments:**

``--server-name <name>``
   Human-readable server name (default: derived from capability name)

``--output, -o <path>``
   Output file path (default: ``./capabilities/<name>.py``)

``--provider <provider>``
   LLM provider override for guide generation (e.g., ``anthropic``, ``openai``, ``cborg``)

``--model <model_id>``
   Model ID override for guide generation (e.g., ``claude-sonnet-4-20250514``, ``gpt-4o``)

``--quiet, -q``
   Reduce output verbosity

**Examples:**

Generate from simulated mode (no server needed):

.. code-block:: bash

   osprey generate capability --from-mcp simulated --name weather_mcp

Generate from real MCP server:

.. code-block:: bash

   osprey generate capability --from-mcp http://localhost:3001 --name slack_mcp

Custom output location:

.. code-block:: bash

   osprey generate capability --from-mcp http://localhost:3001 \
       --name slack_mcp --output ./my_app/capabilities/slack.py

Override LLM provider/model:

.. code-block:: bash

   osprey generate capability --from-mcp simulated --name weather_mcp \
       --provider anthropic --model claude-sonnet-4-20250514

See :doc:`04_mcp-capability-generation` for detailed guide.

osprey generate mcp-server
--------------------------

Generate a demo MCP server for testing Osprey's MCP capability generation. The server uses FastMCP for simple, Pythonic MCP server implementation with weather tools.

**Syntax:**

.. code-block:: bash

   osprey generate mcp-server [OPTIONS]

**Optional Arguments:**

``--name, -n <name>``
   Server name (default: ``demo_mcp``)

``--output, -o <path>``
   Output file path (default: ``./<name>_server.py``)

``--port, -p <port>``
   Server port (default: ``3001``)

**Included Tools:**

The generated server includes three weather-related tools:

- ``get_current_weather`` - Get current weather conditions for a location
- ``get_forecast`` - Get weather forecast for upcoming days
- ``get_weather_alerts`` - Get active weather alerts and warnings

**Examples:**

Generate weather demo server (default):

.. code-block:: bash

   osprey generate mcp-server

Generate on custom port:

.. code-block:: bash

   osprey generate mcp-server --port 3002

Custom output location:

.. code-block:: bash

   osprey generate mcp-server --name my_server --output ./servers/mcp.py

**Running the Generated Server:**

.. code-block:: bash

   # Install FastMCP
   pip install fastmcp

   # Run the server
   python demo_mcp_server.py

**Testing with Osprey:**

.. code-block:: bash

   # Generate capability from the running server
   osprey generate capability --from-mcp http://localhost:3001 --name demo_mcp

See :doc:`04_mcp-capability-generation` for complete workflow.

osprey deploy
================

Manage containerized services (Jupyter, OpenWebUI, Pipelines).

Syntax
------

.. code-block:: bash

   osprey deploy COMMAND [OPTIONS]

Global Options
--------------

``--project PATH`` / ``-p PATH``
   Project directory to use. If not specified, uses ``OSPREY_PROJECT`` environment variable or current directory.

   This option works with all deploy subcommands (``up``, ``down``, ``status``, etc.).

   Example:
      .. code-block:: bash

         osprey deploy status --project ~/projects/my-agent
         osprey deploy up --project ~/projects/my-agent --detached

Commands
--------

``up``
   Start services defined in ``config.yml``.

   Options:
      ``--detached`` - Run services in background

      ``--dev`` - Development mode: use local framework instead of PyPI

   Examples:
      .. code-block:: bash

         osprey deploy up                    # Start in foreground
         osprey deploy up --detached         # Start in background
         osprey deploy up --dev              # Start with local framework
         osprey deploy up --detached --dev   # Background with local framework

``down``
   Stop all running services.

   Example:
      .. code-block:: bash

         osprey deploy down

``restart``
   Restart all services.

   Example:
      .. code-block:: bash

         osprey deploy restart

``status``
   Show status of deployed services.

   Example:
      .. code-block:: bash

         osprey deploy status

``clean``
   Stop services and remove containers and volumes.

   Example:
      .. code-block:: bash

         osprey deploy clean

``rebuild``
   Rebuild containers from scratch (useful after Dockerfile changes).

   Options:
      ``--detached`` - Run services in background after rebuild

      ``--dev`` - Development mode: use local framework instead of PyPI

   Examples:
      .. code-block:: bash

         osprey deploy rebuild                    # Rebuild and start
         osprey deploy rebuild --detached         # Rebuild in background
         osprey deploy rebuild --detached --dev   # Rebuild with local framework

Configuration
-------------

Services are configured in ``config.yml`` under ``deployed_services``:

.. code-block:: yaml

   project_name: "my-agent"  # Project identifier for container tracking

   deployed_services:
     - osprey.jupyter        # Jupyter development environment
     - osprey.open-webui     # Web chat interface
     - osprey.pipelines      # Processing pipeline

**Project Directory:**

All ``osprey deploy`` commands must be run from a project directory (containing ``config.yml``), or use the ``--project`` flag:

.. code-block:: bash

   # Option 1: Run from project directory
   cd my-project
   osprey deploy up

   # Option 2: Use --project flag
   osprey deploy up --project ~/projects/my-project

   # Option 3: Use interactive menu (auto-handles directories)
   osprey

Workflow
--------

**Development workflow:**

.. code-block:: bash

   # Start services in foreground to monitor logs
   osprey deploy up

   # When done, stop with Ctrl+C or:
   osprey deploy down

**Production workflow:**

.. code-block:: bash

   # Start services in background
   osprey deploy up --detached

   # Check status
   osprey deploy status

   # View logs with podman
   podman logs <container_name>

   # Stop when needed
   osprey deploy down

Service Access
--------------

Once deployed, services are available at:

- **OpenWebUI**: http://localhost:8080
- **Jupyter (read-only)**: http://localhost:8088
- **Jupyter (write)**: http://localhost:8089
- **Pipelines**: http://localhost:9099

osprey chat
==============

Start an interactive CLI conversation interface with your agent.

Syntax
------

.. code-block:: bash

   osprey chat [OPTIONS]

Options
-------

``--project PATH`` / ``-p PATH``
   Project directory to use. If not specified, uses ``FRAMEWORK_PROJECT`` environment variable or current directory.

   See :ref:`Global Options <--project>` for multi-project workflow details.

``--config PATH`` / ``-c PATH``
   Path to configuration file.

   Default: ``config.yml`` in project directory

Examples
--------

.. code-block:: bash

   # Start chat in current directory
   osprey chat

   # Start chat in specific project
   osprey chat --project ~/projects/my-agent

   # Use custom config
   osprey chat --config my-config.yml

   # Combine project and config
   osprey chat --project ~/agent --config custom.yml

   # Use environment variable for project
   export OSPREY_PROJECT=~/projects/my-agent
   osprey chat

Usage
-----

The chat interface provides an interactive session with your agent:

.. code-block:: text

   Agent Configuration loaded successfully.
   Registry initialized with 25 capabilities
   ⚡ Use slash commands (/) for quick actions - try /help

   You: What's the weather in San Francisco?

   Agent: [Processing request...]
   The current weather in San Francisco is 18°C with partly cloudy conditions.

Slash Commands
--------------

The CLI supports slash commands for agent control and interface operations:

**Agent Control Commands:**

.. code-block:: bash

   /planning:on          # Enable planning mode
   /planning:off         # Disable planning mode
   /approval:enabled     # Enable approval workflows
   /approval:disabled    # Disable approval workflows
   /approval:selective   # Enable selective approval

**Performance Commands:**

.. code-block:: bash

   /task:off            # Bypass task extraction
   /caps:off            # Bypass capability selection

**CLI Commands:**

.. code-block:: bash

   /help                # Show available commands
   /help <command>      # Show help for specific command
   /exit                # Exit the chat session
   /clear               # Clear the screen

.. seealso::
   :doc:`../../api_reference/01_core_framework/06_command_system`
       Complete API reference for the centralized command system

Prerequisites
-------------

Before using ``framework chat``:

1. Services must be deployed: ``framework deploy up --detached``
2. Configuration must be valid: ``config.yml`` with proper model settings
3. API keys must be set: ``.env`` file with required credentials

osprey config
=============

Manage project configuration settings. All configuration-related operations are unified
under this command group following industry standard CLI patterns (git config, docker config, etc.).

If no subcommand is provided, launches an interactive configuration menu.

Subcommands
-----------

- ``osprey config show`` - Display current project configuration
- ``osprey config export`` - Export framework default configuration
- ``osprey config set-control-system`` - Switch control system connector (mock/epics/tango)
- ``osprey config set-epics-gateway`` - Configure EPICS gateway settings

Syntax
------

.. code-block:: bash

   osprey config [SUBCOMMAND] [OPTIONS]

Examples
--------

**Launch interactive config menu:**

.. code-block:: bash

   osprey config

**Show current configuration:**

.. code-block:: bash

   osprey config show

**Export framework defaults:**

.. code-block:: bash

   osprey config export

**Switch to EPICS:**

.. code-block:: bash

   osprey config set-control-system epics

osprey config show
-------------------

Display current project configuration with syntax highlighting.

Syntax
~~~~~~

.. code-block:: bash

   osprey config show [OPTIONS]

Options
~~~~~~~

``--project PATH`` / ``-p PATH``
   Project directory to use. If not specified, uses current directory or ``OSPREY_PROJECT`` env var.

``--format FORMAT``
   Output format: ``yaml`` (default) or ``json``

Examples
~~~~~~~~

.. code-block:: bash

   # Show current project's config
   osprey config show

   # Show specific project's config
   osprey config show --project ~/my-agent

   # Export as JSON
   osprey config show --format json

osprey config export
---------------------

Export the Osprey framework's default configuration template.

This shows the complete framework template with all available options and default values.
Useful for understanding what configuration options are available.

Syntax
~~~~~~

.. code-block:: bash

   osprey config export [OPTIONS]

Options
~~~~~~~

``--output PATH`` / ``-o PATH``
   Save configuration to file instead of printing to console.

``--format FORMAT``
   Output format: ``yaml`` (default) or ``json``

Examples
~~~~~~~~

.. code-block:: bash

   # Display to console
   osprey config export

   # Save to file
   osprey config export -o defaults.yml

   # Export as JSON
   osprey config export --format json -o defaults.json

   # Use as reference when customizing
   osprey config export --output reference.yml
   diff reference.yml config.yml

osprey config set-control-system
----------------------------------

Switch control system connector type (mock, epics, tango, labview).

This changes the ``control_system.type`` setting in config.yml, which determines
which connector is used at runtime for control system operations.

.. note::
   Pattern detection is control-system-agnostic. This setting only affects which
   connector is loaded at runtime, not which patterns are used for security detection.

Syntax
~~~~~~

.. code-block:: bash

   osprey config set-control-system SYSTEM_TYPE [OPTIONS]

Arguments
~~~~~~~~~

``SYSTEM_TYPE``
   Control system type: ``mock``, ``epics``, ``tango``, or ``labview``

Options
~~~~~~~

``--project PATH`` / ``-p PATH``
   Project directory to use. If not specified, uses current directory.

Examples
~~~~~~~~

.. code-block:: bash

   # Switch to mock mode (development)
   osprey config set-control-system mock

   # Switch to EPICS (production)
   osprey config set-control-system epics

   # Switch to Tango
   osprey config set-control-system tango

osprey config set-epics-gateway
---------------------------------

Configure EPICS gateway address and port settings.

Can use facility presets (ALS, APS) or specify custom gateway settings.

Syntax
~~~~~~

.. code-block:: bash

   osprey config set-epics-gateway [OPTIONS]

Options
~~~~~~~

``--facility FACILITY``
   Facility preset: ``als``, ``aps``, or ``custom``

``--address ADDRESS``
   Gateway address (required for custom facility)

``--port PORT``
   Gateway port (required for custom facility)

``--project PATH`` / ``-p PATH``
   Project directory to use. If not specified, uses current directory.

Examples
~~~~~~~~

.. code-block:: bash

   # Use ALS gateway preset
   osprey config set-epics-gateway --facility als

   # Use APS gateway preset
   osprey config set-epics-gateway --facility aps

   # Set custom gateway
   osprey config set-epics-gateway --facility custom \
       --address gateway.example.com --port 5064

============

osprey export-config (DEPRECATED)
===================================

.. deprecated::
   Use ``osprey config export`` instead. This command is kept for backward compatibility
   but will be removed in a future version.

Export the framework's default configuration for reference.

Syntax
------

.. code-block:: bash

   osprey export-config [OPTIONS]

Migration
---------

Replace ``osprey export-config`` with ``osprey config export``:

.. code-block:: bash

   # Old (deprecated)
   osprey export-config
   osprey export-config -o file.yml

   # New (recommended)
   osprey config export
   osprey config export -o file.yml

Use Cases
---------

1. **Discover available options** - See all configuration fields and their defaults
2. **Reference template** - Use as starting point for custom configurations
3. **Troubleshooting** - Compare your config with framework defaults
4. **Documentation** - Understand configuration structure

Configuration Structure
-----------------------

The exported configuration includes:

- **Models**: LLM provider configurations (orchestrator, classifier, code generator)
- **Services**: Jupyter, OpenWebUI, Pipelines settings
- **Execution Control**: Timeouts, retry policies, safety limits
- **File Paths**: Directory structures and artifact locations
- **Logging**: Log levels and output settings


Interactive Configuration
=========================

.. admonition:: New in v0.9.6: Interactive Configuration Management
   :class: version-09plus-change

   The interactive menu now includes a configuration submenu for managing project settings.
   Access it via: **Project Menu** → ``config`` → Choose action

Available Configuration Actions
-------------------------------

When you select ``config`` from the project menu, you get access to:

**1. Show Configuration**
   Display current project configuration (equivalent to ``osprey config show``)

**2. Set Control System Type**
   Switch between Mock (tutorial/development) and EPICS (production) connectors

   - Automatically updates ``control_system.type``
   - Optionally updates ``archiver.type``
   - Shows current configuration before changes
   - Provides next-step guidance

**3. Set EPICS Gateway**
   Configure EPICS gateway for production deployment

   - **APS** preset: ``pvgatemain1.aps4.anl.gov:5064``
   - **ALS** preset: ``cagw-alsdmz.als.lbl.gov:5064`` (read), ``5084`` (write)
   - **Custom**: Interactive prompts for your facility
   - Automatically detects current facility configuration

Example Workflow
----------------

**Tutorial → Production Migration:**

.. code-block:: text

   1. Create project (starts in Mock mode by default)
      $ osprey init my-control-assistant --template control_assistant

   2. Develop with Mock data (no hardware needed)
      $ osprey chat
      You: "What is the beam current?"

   3. When ready for production, launch interactive menu:
      $ osprey

   4. Select your project → config → set-control-system
      → Choose: EPICS - Production mode
      → Choose: Yes - Use EPICS Archiver Appliance
      → Confirm changes

   5. Configure EPICS gateway:
      config → set-epics-gateway
      → Choose: ALS (or APS, or Custom)
      → Confirm changes

   6. Test production connection:
      $ osprey chat
      You: "What is the beam current?"

**What Changes Under the Hood:**

The interactive commands update your ``config.yml``:

.. code-block:: yaml

   # Before (Tutorial Mode)
   control_system:
     type: mock
   archiver:
     type: mock_archiver

   # After (Production Mode)
   control_system:
     type: epics
     connector:
       epics:
         gateways:
           read_only:
             address: cagw-alsdmz.als.lbl.gov  # From facility preset
             port: 5064
   archiver:
     type: epics_archiver

Your capabilities work unchanged - ``ConnectorFactory`` automatically uses the configured connector.

**See Also:**

- :ref:`Migrate to Production Control System <migrate-to-production>` in Control Assistant Part 3
- :doc:`../05_production-systems/06_control-system-integration` for connector architecture


Environment Variables
=====================

The framework uses environment variables for API keys, paths, and deployment-specific configuration.

For a **complete list of all supported environment variables** with descriptions and examples, see the :ref:`Environment Variables section <environment-variables>` in the Installation Guide.

**Quick Reference:**

.. code-block:: bash

   # Required
   PROJECT_ROOT=/path/to/your/project
   OPENAI_API_KEY=sk-...          # Or ANTHROPIC_API_KEY, GOOGLE_API_KEY, CBORG_API_KEY

   # Optional - Multi-project support (New in v0.7.7)
   OSPREY_PROJECT=/path/to/project

   # Optional - Other settings
   LOCAL_PYTHON_VENV=/path/to/venv
   TZ=America/Los_Angeles
   CONFIG_FILE=custom-config.yml

   # Proxy settings (if needed)
   HTTP_PROXY=http://proxy:8080
   NO_PROXY=localhost,127.0.0.1

``OSPREY_PROJECT``
   Default project directory for all commands. Allows working with a specific project from any location without using the ``--project`` flag on every command.

   **Priority:** Lower than ``--project`` flag, higher than current directory.

   **Example:**

   .. code-block:: bash

      export OSPREY_PROJECT=~/projects/my-agent
      osprey chat           # Uses ~/projects/my-agent
      osprey deploy status  # Uses ~/projects/my-agent

Common Workflows
================

Complete Project Setup
----------------------

.. code-block:: bash

   # 1. Install framework
   pip install osprey-framework

   # 2. Create project
   osprey init weather-agent --template hello_world_weather
   cd weather-agent

   # 3. Configure environment
   cp .env.example .env
   # Edit .env with your API keys

   # 4. Update config (optional)
   # Edit config.yml as needed

   # 5. Deploy services
   osprey deploy up --detached

   # 6. Start chat
   osprey chat

Development Workflow
--------------------

.. code-block:: bash

   # Start services for development
   osprey deploy up

   # In another terminal, make changes to your code
   # Test with chat interface
   osprey chat

   # Rebuild containers if needed
   osprey deploy rebuild

   # Clean up
   osprey deploy clean

Framework Development Workflow
------------------------------

If you're developing the framework itself:

.. code-block:: bash

   # Start services with local framework
   osprey deploy up --dev

   # Make changes to framework code
   # Rebuild to test changes
   osprey deploy rebuild --dev

   # Verify local framework is used
   podman exec jupyter-read pip show osprey

Multi-Project Workflows
-----------------------

.. admonition:: New in v0.7.7: Multi-Project Support
   :class: version-07plus-change

   Work with multiple projects simultaneously using the ``--project`` flag or ``FRAMEWORK_PROJECT`` environment variable.

**Scenario 1: Parallel Development**

Work on multiple projects from a central location:

.. code-block:: bash

   # Check status of all projects
   osprey deploy status --project ~/projects/weather-agent
   osprey deploy status --project ~/projects/turbine-monitor
   osprey deploy status --project ~/projects/als-assistant

   # Start chat with specific project
   osprey chat --project ~/projects/weather-agent

**Scenario 2: Dedicated Terminal per Project**

Use environment variables for persistent project selection:

.. code-block:: bash

   # Terminal 1: Weather Agent
   export OSPREY_PROJECT=~/projects/weather-agent
   osprey deploy up --detached
   osprey chat

   # Terminal 2: Turbine Monitor
   export OSPREY_PROJECT=~/projects/turbine-monitor
   osprey deploy up --detached
   osprey health

   # Terminal 3: Jump between projects
   osprey chat --project ~/projects/weather-agent
   osprey chat --project ~/projects/turbine-monitor

**Scenario 3: CI/CD Pipeline**

Automate deployment and testing across multiple projects:

.. code-block:: bash

   #!/bin/bash
   # Deploy and test multiple projects

   PROJECTS=(
       ~/projects/weather-agent
       ~/projects/turbine-monitor
       ~/projects/als-assistant
   )

   for project in "${PROJECTS[@]}"; do
       echo "Deploying $project..."
       osprey deploy up --detached --project "$project"
       osprey health --project "$project"
   done

**Scenario 4: Development + Production**

Work with development and production configurations:

.. code-block:: bash

   # Development environment
   export OSPREY_PROJECT=~/dev/my-agent
   osprey deploy up

   # In another terminal, check production
   osprey deploy status --project /opt/production/my-agent

Configuration Reference
-----------------------

.. code-block:: bash

   # View framework defaults
   osprey config export

   # Export to file for reference
   osprey config export --output defaults.yml

   # Create new project and compare configs
   osprey init test-project
   diff defaults.yml test-project/config.yml

Troubleshooting
===============

Command Not Found
-----------------

If ``osprey`` command is not found:

.. code-block:: bash

   # Verify installation
   pip show osprey-framework

   # Reinstall if needed
   pip install --upgrade osprey-framework

   # Check pip bin directory is in PATH
   python -m pip show osprey-framework

Services Won't Start
--------------------

.. code-block:: bash

   # Check podman is running
   podman --version
   podman ps

   # Check for port conflicts
   lsof -i :8080
   lsof -i :9099

   # Try starting services in foreground to see errors
   osprey deploy up

Configuration Errors
--------------------

.. code-block:: bash

   # Validate against framework defaults
   osprey config export --output defaults.yml

   # Check your config syntax
   cat config.yml

   # Ensure environment variables are set
   cat .env

Chat Not Responding
-------------------

.. code-block:: bash

   # Verify services are running
   osprey deploy status
   podman ps

   # Check API keys are set
   cat .env

   # Verify model configuration
   grep -A 10 "models:" config.yml

.. seealso::

   :doc:`../01_understanding-the-framework/02_convention-over-configuration`
       Framework architecture and conventions

   :doc:`../../getting-started/installation`
       Installation and setup guide

   :doc:`../05_production-systems/05_container-and-deployment`
       Container deployment details



