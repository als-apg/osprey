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
   - Managing configuration with ``osprey config``
   - Running interactive sessions with ``osprey chat``
   - Managing deployments with ``osprey deploy``
   - Generating capabilities from MCP servers with ``osprey generate``
   - Generating soft IOCs for testing with ``osprey generate soft-ioc``

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
   osprey config             # Manage configuration
   osprey chat               # Start interactive chat
   osprey deploy COMMAND     # Manage services
   osprey generate COMMAND   # Generate components (MCP capabilities, servers)
   osprey channel-finder     # Channel finder CLI (query, benchmark, interactive)
   osprey eject              # Copy framework components for customization
   osprey tasks              # Browse AI assistant tasks (NEW)
   osprey claude             # Manage Claude Code skills (NEW)

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

The interactive flow is equivalent to using ``osprey init`` with appropriate flags, but with helpful guidance and validation at each step.

Disabling Interactive Mode
--------------------------

If you prefer to only use direct commands, you can bypass the interactive menu by:

- Running specific commands directly: ``osprey chat``, ``osprey deploy up``, etc.
- Using ``osprey --help`` to see available commands
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
2. **``OSPREY_PROJECT`` environment variable**
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

   - ``extend`` - Use helper functions (5-10 lines, recommended)
   - ``standalone`` - Full registry implementation (verbose, for learning)

   Default: ``extend``

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

   osprey init my-agent --template minimal --registry-style standalone

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
   │       ├── framework_prompts/  # Prompt customizations
   │       └── capabilities/   # Agent capabilities
   ├── services/               # Container services
   ├── config.yml             # Complete configuration
   ├── .env.example           # Environment template
   └── README.md              # Project documentation

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
- ``osprey config set-models`` - Configure AI provider and models for all model roles

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

**Configure AI models:**

.. code-block:: bash

   osprey config set-models

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
-------------------------------

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

osprey config set-models
------------------------

Configure AI provider and models for all model roles.

Updates ALL model configurations in config.yml to use the specified provider
and model. This includes orchestrator, response, classifier, and any custom
models defined in your project (e.g., channel_write, channel_finder).

The max_tokens settings for each model role will be preserved.

If no options are provided, launches an interactive selection menu.

Syntax
~~~~~~

.. code-block:: bash

   osprey config set-models [OPTIONS]

Options
~~~~~~~

``--provider PROVIDER``
   AI provider: ``anthropic``, ``openai``, ``google``, ``cborg``, ``ollama``, or ``amsc``

``--model MODEL``
   Model identifier (e.g., ``claude-sonnet-4``, ``gpt-4``, ``anthropic/claude-haiku``)

``--project PATH`` / ``-p PATH``
   Project directory to use. If not specified, uses current directory.

Examples
~~~~~~~~

.. code-block:: bash

   # Interactive mode (recommended)
   osprey config set-models

   # Set all models to Anthropic Claude
   osprey config set-models --provider anthropic --model claude-sonnet-4

   # Set all models to CBORG provider for specific project
   osprey config set-models --provider cborg --model anthropic/claude-haiku --project ~/my-agent

osprey chat
==============

Start an interactive conversation interface with your agent.

Syntax
------

.. code-block:: bash

   osprey chat [OPTIONS]

Options
-------

``--tui``
   Launch the Terminal User Interface (TUI) instead of the default CLI.

   .. admonition:: Experimental Feature (New in v0.10.0)
      :class: warning

      The TUI is an experimental feature available for testing. It provides a full-screen
      terminal experience with real-time streaming and visual step tracking.

   **Requirements:** ``pip install osprey-framework[tui]``

``--project PATH`` / ``-p PATH``
   Project directory to use. If not specified, uses ``OSPREY_PROJECT`` environment variable or current directory.

   See :ref:`Global Options <--project>` for multi-project workflow details.

``--config PATH`` / ``-c PATH``
   Path to configuration file.

   Default: ``config.yml`` in project directory

Examples
--------

.. code-block:: bash

   # Start CLI chat (default)
   osprey chat

   # Start TUI chat (experimental)
   osprey chat --tui

   # Start chat in specific project
   osprey chat --project ~/projects/my-agent

   # TUI with specific project
   osprey chat --tui --project ~/projects/my-agent

   # Use custom config
   osprey chat --config my-config.yml

   # Use environment variable for project
   export OSPREY_PROJECT=~/projects/my-agent
   osprey chat

Terminal User Interface (TUI)
-----------------------------

.. admonition:: Experimental Feature (New in v0.10.0)
   :class: warning

   The TUI is experimental and available for testing. Feedback welcome!

The TUI provides a full-screen terminal experience built with `Textual <https://textual.textualize.io/>`_:

**Features:**

- **Real-time Streaming**: Watch agent responses appear character-by-character
- **Step Visualization**: See Task Extraction → Classification → Orchestration → Execution in real-time
- **15+ Built-in Themes**: Switch themes instantly with ``Ctrl+T``
- **Command Palette**: Quick access to all actions with ``Ctrl+P``
- **Slash Commands**: ``/exit``, ``/caps:on``, ``/caps:off``, and more
- **Query History**: Navigate previous queries with up/down arrows
- **Content Viewer**: Multi-tab view for prompts and responses
- **Todo Visualization**: See agent planning progress

**Keyboard Shortcuts:**

.. list-table::
   :widths: 20 40
   :header-rows: 1

   * - Shortcut
     - Action
   * - ``Ctrl+P``
     - Open command palette
   * - ``Ctrl+T``
     - Open theme picker
   * - ``Ctrl+L``
     - Focus input
   * - ``Ctrl+H``
     - Toggle help panel
   * - ``Ctrl+C`` (twice)
     - Exit TUI
   * - ``Space``/``b``
     - Scroll down/up
   * - ``g``/``G``
     - Go to top/bottom

**Installation:**

.. code-block:: bash

   pip install osprey-framework[tui]

**Interactive Menu:**

The TUI is also accessible from the interactive menu as "chat (tui)"

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

**Direct Chat Mode Commands:**

.. code-block:: bash

   /chat                     # List capabilities that support direct chat
   /chat:<capability_name>   # Enter direct chat mode with a specific capability
   /exit                     # Exit direct chat mode (or exit CLI if not in direct chat)

**CLI Commands:**

.. code-block:: bash

   /help                # Show available commands
   /help <command>      # Show help for specific command
   /exit                # Exit direct chat mode (or exit CLI if not in direct chat)
   /clear               # Clear the screen

.. _direct-chat-mode:

Direct Chat Mode
----------------

Direct Chat Mode enables multi-turn conversations directly with a specific capability, bypassing the normal orchestration pipeline (task extraction → classification → orchestration). This is useful for:

- **Interactive exploration** with ReAct-style capabilities
- **Focused conversations** where you know which capability you need
- **Context accumulation** across multiple turns within the same capability

**Available Capabilities:**

Direct chat mode is designed for **ReAct-style capabilities** - agents that use tools and benefit from multi-turn reasoning. The framework includes one built-in direct-chat capability:

- ``state_manager`` - Inspect and manage accumulated context data

You can create your own ReAct capabilities with direct chat support. One example is generating a capability from an MCP server - see :doc:`04_mcp-capability-generation` for a tutorial that creates the ``weather_mcp`` capability shown in these examples.

**Entering Direct Chat Mode:**

.. code-block:: text

   👤 You: /chat
   Available capabilities for direct chat:
   ┌──────────────────┬─────────────────────────────────────┐
   │ Capability       │ Description                         │
   ├──────────────────┼─────────────────────────────────────┤
   │ state_manager    │ Manage and inspect agent state      │
   │ weather_mcp      │ Weather operations via MCP server   │
   └──────────────────┴─────────────────────────────────────┘

   👤 You: /chat:weather_mcp
   ✓ Entering direct chat with weather_mcp
     Type /exit to return to normal mode

   🎯 weather_mcp > What's the weather in Tokyo?
   🤖 The current weather in Tokyo is 22°C with clear skies...

   🎯 weather_mcp > How about San Francisco?
   🤖 San Francisco is currently 18°C with partly cloudy conditions...

.. note::

   The ``weather_mcp`` capability shown above is an example generated from an MCP server. Your ``/chat`` list will only show ``state_manager`` until you generate or create additional direct-chat-enabled capabilities.

**Key Behaviors:**

- **Message history preserved**: The capability sees the full conversation history, enabling follow-up questions like "How about yesterday?" or "Compare that to Boston"
- **Pipeline bypass**: Messages go directly to the capability without task extraction, classification, or orchestration
- **Visual indicator**: The prompt changes to show the active capability (e.g., ``🎯 weather_mcp >``)

**Saving Results to Context:**

While in direct chat mode, you can save results for later use in orchestrated queries:

.. code-block:: text

   🎯 weather_mcp > What's the weather in Tokyo?
   🤖 Tokyo is 22°C with clear skies...

   🎯 weather_mcp > Save that as tokyo_weather
   🤖 ✓ Saved weather data as 'tokyo_weather'

   🎯 weather_mcp > /exit
   ✓ Exited direct chat with weather_mcp

   👤 You: Compare the tokyo_weather to current Boston conditions
   🤖 [Orchestrated query using saved context...]

**State Manager Capability:**

The built-in ``state_manager`` capability provides tools for inspecting and managing accumulated context:

.. code-block:: text

   👤 You: /chat:state_manager
   ✓ Entering direct chat with state_manager

   🎯 state_manager > What context data do we have?
   🤖 Current context includes:
      - WEATHER_RESULTS: tokyo_weather, sf_weather
      - ANALYSIS_RESULTS: correlation_analysis

   🎯 state_manager > Show me the tokyo_weather details
   🤖 [Displays full context object...]

**Exiting Direct Chat Mode:**

Use ``/exit`` to return to normal orchestrated mode:

.. code-block:: text

   🎯 weather_mcp > /exit
   ✓ Exited direct chat with weather_mcp
     Returning to normal mode

   👤 You: [Now in normal orchestrated mode]

.. note::

   Not all capabilities support direct chat mode. Only capabilities with ``direct_chat_enabled = True`` appear in the ``/chat`` list. See :doc:`01_building-your-first-capability` for how to enable this on your own capabilities.

.. seealso::
   :doc:`../../api_reference/01_core_framework/06_command_system`
       Complete API reference for the centralized command system

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

osprey generate
===============

Generate Osprey components from various sources.

.. note::
   All ``osprey generate`` subcommands are **prototype features** under active development.

Subcommands
-----------

``osprey generate capability``
   Generate Osprey capability from MCP server or natural language prompt

``osprey generate mcp-server``
   Generate demo MCP server for testing

``osprey generate soft-ioc``
   Generate Python soft IOC for EPICS testing (caproto-based)

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
   LLM provider override for guide generation (e.g., ``anthropic``, ``openai``, ``cborg``, ``amsc``)

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

.. _cli-generate-soft-ioc:

osprey generate soft-ioc
------------------------

Generate a pure Python EPICS soft IOC for testing control system integrations without requiring real hardware or an EPICS base installation.

**Syntax:**

.. code-block:: bash

   osprey generate soft-ioc [OPTIONS]

**Optional Arguments:**

``--config, -c <path>``
   Config file path (default: ``config.yml``)

``--output, -o <path>``
   Override output file path

``--dry-run``
   Preview generation without writing files

``--init``
   Force interactive setup (creates/overwrites simulation config)

**Configuration:**

The command reads from the ``simulation`` section in ``config.yml``:

.. code-block:: yaml

   simulation:
     channel_database: "data/channel_databases/hierarchical.json"
     pairings_file: "data/pairings.json"  # Optional SP/RB mappings
     ioc:
       name: "accelerator_sim"
       port: 5064
       output_dir: "generated_iocs/"
     base:                     # Single dict (no dash) - the foundation
       type: "mock_style"      # Built-in backend
       noise_level: 0.01       # 1% noise for SP→RB tracking
       update_rate: 10.0       # Simulation update rate in Hz
     overlays: []              # List (optional) - override behaviors

**Simulation Backends:**

Two built-in backends are available:

- ``mock_style`` (recommended): Archiver-style simulation with SP→RB tracking and
  PV-type-specific behaviors based on naming conventions:

  - **BPM/position PVs**: Random equilibrium offset with slow drift and noise
  - **Beam current PVs**: 500 mA base with decay over time
  - **Voltage PVs**: 5000 V base, stable with small oscillation
  - **Pressure PVs**: 1e-9 Torr base with gradual increase
  - **Temperature PVs**: 25°C base with gradual increase

- ``passthrough``: No simulation - PVs retain written values without automatic updates.
  Useful for manual testing or when you want full control over PV values.

Multiple backends can be chained using ``base`` + ``overlays``:

.. code-block:: yaml

   simulation:
     base:
       type: "mock_style"                        # Base backend (defaults)
     overlays:
       - module_path: "my_tests.fault_backends"  # Override backend
         class_name: "BrokenFeedbackBackend"
         params:
           target_pv: "QUAD:Q1:CURRENT"

**Key Features:**

- Supports all 4 Osprey channel database types (flat, template, hierarchical, middle_layer)
- Auto-detects database type from file structure
- Infers PV types and access modes from naming conventions (SP/RB, STATUS, IMAGE, etc.)
- Generates ``SIM:HEARTBEAT`` PV for monitoring IOC health
- Interactive setup wizard if no simulation config exists
- Offers to update ``config.yml`` to connect to the generated IOC

**Examples:**

Interactive setup (recommended for first use):

.. code-block:: bash

   osprey generate soft-ioc --init

Preview what would be generated:

.. code-block:: bash

   osprey generate soft-ioc --dry-run

Generate using existing config:

.. code-block:: bash

   osprey generate soft-ioc

**Workflow:**

.. code-block:: bash

   # 1. Generate the IOC (interactive setup if needed)
   osprey generate soft-ioc --init

   # 2. Install numpy (caproto is included with osprey-framework)
   pip install numpy

   # 3. Run the IOC
   python generated_iocs/accelerator_sim_ioc.py

   # 4. Verify it's running (heartbeat increments every 100ms)
   caget SIM:HEARTBEAT

   # 5. Test with Osprey chat
   osprey chat
   You: "Show me all quadrupole currents"

**Connecting to Generated IOC:**

After generation, the command offers to update ``config.yml`` to connect to the soft IOC.
You can also configure manually:

.. code-block:: yaml

   control_system:
     type: epics
     connector:
       epics:
         gateways:
           read_only:
             address: localhost
             port: 5064

Or use the "Local Simulation" facility preset via the interactive menu or:

.. code-block:: bash

   osprey config set-epics-gateway --facility simulation

**SP/RB Pairings:**

For setpoint-readback tracking, create a JSON file mapping setpoint PV names to their
corresponding readback PVs:

.. code-block:: json

   {
     "QUAD:CURRENT:SP": "QUAD:CURRENT:RB",
     "DIPOLE:FIELD:SP": "DIPOLE:FIELD:RB"
   }

When a setpoint is written, the ``mock_style`` backend automatically updates the
paired readback with configurable noise (default 1%).

**Custom Backends:**

Create custom backends for physics simulation or fault injection by implementing
the ``SimulationBackend`` protocol:

.. code-block:: python

   class BrokenFeedbackBackend:
       """Simulates broken feedback: SP writes don't affect RB."""

       def __init__(self, target_pv: str, drift_rate: float = 0.5):
           self.sp = f"{target_pv}:SP"
           self.rb = f"{target_pv}:RB"
           self.drift_rate = drift_rate
           self._rb_value = 100.0

       def initialize(self, pv_definitions):
           return {self.rb: self._rb_value}

       def on_write(self, pv_name, value):
           if pv_name == self.sp:
               return {}  # Block normal SP->RB update
           return None  # Delegate to next backend

       def step(self, dt):
           self._rb_value += self.drift_rate * dt
           return {self.rb: self._rb_value}

Configure in ``config.yml``:

.. code-block:: yaml

   simulation:
     base:
       type: "mock_style"                        # Base
     overlays:
       - module_path: "my_tests.fault_backends"  # Override
         class_name: "BrokenFeedbackBackend"
         params:
           target_pv: "QUAD:Q1:CURRENT"
           drift_rate: 0.5

.. seealso::

   :doc:`05_soft-ioc-backends`
       Complete guide to implementing custom physics simulation backends (pyAT, OCELOT),
       the SimulationBackend Protocol, and chained backend composition.

=====================

osprey channel-finder
=====================

Natural language channel search tool. Provides interactive REPL mode, direct queries, and benchmarking for evaluating channel finder performance.

Syntax
------

.. code-block:: bash

   osprey channel-finder [OPTIONS] [COMMAND]

Options
-------

``--project`` / ``-p``
   Project directory (default: current directory or ``OSPREY_PROJECT`` env var)

``--verbose`` / ``-v``
   Enable verbose logging

Commands
--------

``osprey channel-finder`` (no subcommand)
   Launch interactive REPL for channel finding queries. Type queries in natural language
   and see matched channels in real time.

``osprey channel-finder query "QUERY_TEXT"``
   Execute a single channel finder query and display results.

``osprey channel-finder benchmark``
   Run channel finder benchmarks against benchmark datasets. Results are saved
   to ``data/benchmarks/results/``.

   Options:

   ``--queries``
      Query selection (e.g., ``"all"``, ``"0:10"``, ``"0,5,10"``)

   ``--model``
      Override model (e.g., ``anthropic/claude-sonnet``)

   ``--dataset``
      Path to custom benchmark dataset JSON file

   ``--verbose`` / ``-v``
      Show detailed channel finder logs

``osprey channel-finder build-database``
   Build a channel database from a CSV file. Reads CSV with columns:
   ``address``, ``description``, ``family_name``, ``instances``, ``sub_channel``.
   Rows with ``family_name`` are grouped into templates; rows without are standalone.

   Options:

   ``--csv PATH``
      Input CSV file (default: ``data/raw/address_list.csv``)

   ``--output PATH``
      Output JSON file (default: ``data/processed/channel_database.json``)

   ``--use-llm``
      Use LLM to generate descriptive names for standalone channels

   ``--config PATH``
      Path to facility config file (optional, auto-detected)

   ``--delimiter CHAR``
      CSV field delimiter (default: ``,``). Use ``|`` or ``\t`` if channel names contain commas.

``osprey channel-finder validate``
   Validate a channel database JSON file. Checks JSON structure, schema validity,
   and database loading. Auto-detects pipeline type (hierarchical vs in_context).

   Options:

   ``--database PATH`` / ``-d PATH``
      Path to database file (default: from config)

   ``--verbose`` / ``-v``
      Show detailed statistics

   ``--pipeline``
      Override pipeline type detection: ``hierarchical`` or ``in_context``

``osprey channel-finder preview``
   Preview a channel database with flexible display options. Auto-detects database
   type (hierarchical, in_context, middle_layer) and shows a tree visualization.

   Options:

   ``--depth N``
      Tree depth to display (default: 3, use -1 for unlimited)

   ``--max-items N``
      Maximum items per level (default: 3, use -1 for unlimited)

   ``--sections SECTIONS``
      Comma-separated sections: ``tree``, ``stats``, ``breakdown``, ``samples``, ``all`` (default: ``tree``)

   ``--focus PATH``
      Focus on specific path (e.g., ``"M:QB"`` for QB family in M system)

   ``--database PATH``
      Direct path to database file (overrides config, auto-detects type)

   ``--full``
      Show complete hierarchy (shorthand for ``--depth -1 --max-items -1``)

Examples
--------

.. code-block:: bash

   # Interactive REPL (default)
   osprey channel-finder

   # Direct query
   osprey channel-finder query "find beam position monitors"

   # Run all benchmarks
   osprey channel-finder benchmark

   # Benchmark subset with specific model
   osprey channel-finder benchmark --queries 0:10 --model anthropic/claude-sonnet

   # Build database from CSV
   osprey channel-finder build-database --csv data/raw/channels.csv

   # Build with LLM-generated names
   osprey channel-finder build-database --csv data/raw/channels.csv --use-llm

   # Validate configured database
   osprey channel-finder validate

   # Validate specific file
   osprey channel-finder validate --database data/processed/db.json --verbose

   # Preview database (quick overview)
   osprey channel-finder preview

   # Preview with stats and full tree
   osprey channel-finder preview --depth 4 --sections tree,stats

   # Preview specific database file
   osprey channel-finder preview --database data/processed/db.json --full

   # Use with specific project
   osprey channel-finder --project ~/my-agent query "vacuum gauges"

.. _cli-eject:

============

osprey eject
============

Copy framework-native capabilities or services to your local project for customization. Use this when you need to modify framework behavior beyond what prompt customization allows.

Syntax
------

.. code-block:: bash

   osprey eject COMMAND [OPTIONS]

Commands
--------

``osprey eject list``
   List all ejectable framework capabilities and services.

``osprey eject capability NAME``
   Copy a framework capability to your local project for customization.

   Options:

   ``--output`` / ``-o``
      Output file path (default: ``./src/<package>/capabilities/<name>.py``)

   ``--include-tests``
      Also copy related test files

``osprey eject service NAME``
   Copy a framework service (entire directory) to your local project for customization.

   Options:

   ``--output`` / ``-o``
      Output directory path (default: ``./src/<package>/services/<name>/``)

   ``--include-tests``
      Also copy related test files

Available Components
--------------------

**Capabilities:**

- ``channel_finding`` — Find control system channels using semantic search
- ``channel_read`` — Read current values from control system channels
- ``channel_write`` — Write values to control system channels
- ``archiver_retrieval`` — Query historical time-series data from archivers

**Services:**

- ``channel_finder`` — Semantic channel finding service (pipelines, databases, benchmarks)

Examples
--------

.. code-block:: bash

   # List all ejectable components
   osprey eject list

   # Copy channel finding capability to local project
   osprey eject capability channel_finding

   # Copy entire channel finder service
   osprey eject service channel_finder

   # Copy with tests
   osprey eject capability channel_finding --include-tests

   # Custom output location
   osprey eject capability channel_finding -o ./src/my_app/capabilities/my_channel_finding.py

**After Ejecting:**

1. Modify the ejected files for your needs
2. Register the local version in your registry using ``override_capabilities``
3. Run ``osprey health`` to verify the configuration

============

osprey ariel
============

Manage the ARIEL (Agentic Retrieval Interface for Electronic Logbooks) search service.
Commands for database setup, data ingestion, search, embedding management, and the web interface.

Syntax
------

.. code-block:: bash

   osprey ariel COMMAND [OPTIONS]

Commands
--------

``osprey ariel quickstart``
   Run the complete setup sequence: check database, migrate, and ingest demo data.

   Options:
      ``-s, --source PATH`` — Custom logbook JSON file (default: config or bundled demo data)

   .. code-block:: bash

      osprey ariel quickstart                      # Use config defaults
      osprey ariel quickstart -s my_logbook.json   # Custom data source

``osprey ariel status``
   Show ARIEL service status including database connection, embedding tables, and module states.

   Options:
      ``--json`` — Output as JSON

   .. code-block:: bash

      osprey ariel status          # Human-readable output
      osprey ariel status --json   # JSON output

``osprey ariel migrate``
   Create or update database tables based on enabled modules.

   Options:
      ``--rollback`` — Rollback migrations (not yet implemented)

   .. code-block:: bash

      osprey ariel migrate

``osprey ariel ingest``
   Ingest logbook entries from a source file or URL.

   Options:
      ``-s, --source`` (required) — Source file path or URL
      ``-a, --adapter`` — Adapter type: ``als_logbook``, ``jlab_logbook``, ``ornl_logbook``, ``generic_json`` (default: ``generic_json``)
      ``--since`` — Only ingest entries after this date
      ``--limit`` — Maximum entries to ingest
      ``--dry-run`` — Parse and count entries without storing

   .. code-block:: bash

      osprey ariel ingest -s data/logbook.json                    # Generic JSON
      osprey ariel ingest -s data/logbook.json -a als_logbook     # ALS adapter
      osprey ariel ingest -s data/logbook.json --since 2024-01-01 # Filter by date
      osprey ariel ingest -s data/logbook.json --dry-run          # Parse only

``osprey ariel watch``
   Watch a source for new logbook entries. Continuously polls and ingests new entries.

   Options:
      ``-s, --source`` — Source file path or URL (overrides config)
      ``-a, --adapter`` — Adapter type (overrides config)
      ``--once`` — Run a single poll cycle and exit
      ``--interval`` — Override poll interval in seconds
      ``--dry-run`` — Show what would be ingested without storing

   .. code-block:: bash

      osprey ariel watch                           # Watch using config
      osprey ariel watch --once --dry-run          # Preview one cycle
      osprey ariel watch --interval 300            # Poll every 5 minutes
      osprey ariel watch -s https://api/logbook    # Override source URL

``osprey ariel enhance``
   Run enhancement modules on ingested entries.

   Options:
      ``-m, --module`` — Specific module: ``text_embedding`` or ``semantic_processor``
      ``--force`` — Re-process already enhanced entries
      ``--limit`` — Maximum entries to process (default: 100)

   .. code-block:: bash

      osprey ariel enhance                           # Run all enabled modules
      osprey ariel enhance -m text_embedding         # Run specific module
      osprey ariel enhance --force --limit 500       # Re-process up to 500 entries

``osprey ariel models``
   List embedding models and their database tables.

   .. code-block:: bash

      osprey ariel models

``osprey ariel search``
   Execute a search query from the command line.

   Options:
      ``QUERY`` (required argument) — Search query text
      ``--mode`` — Search mode: ``keyword``, ``semantic``, ``rag``, ``auto`` (default: ``auto``)
      ``--limit`` — Maximum results (default: 10)
      ``--json`` — Output as JSON

   .. code-block:: bash

      osprey ariel search "RF cavity fault"                # Auto mode
      osprey ariel search "beam loss" --mode keyword       # Keyword only
      osprey ariel search "what caused the trip?" --mode rag
      osprey ariel search "RF" --limit 5 --json            # JSON output

``osprey ariel reembed``
   Re-embed entries with a new or existing model. Creates the embedding table if needed.

   Options:
      ``--model`` (required) — Embedding model name (e.g., ``nomic-embed-text``)
      ``--dimension`` (required) — Embedding dimension (e.g., 768)
      ``--batch-size`` — Entries per batch (default: 100)
      ``--dry-run`` — Show what would be done
      ``--force`` — Overwrite existing embeddings

   .. code-block:: bash

      osprey ariel reembed --model nomic-embed-text --dimension 768
      osprey ariel reembed --model mxbai-embed-large --dimension 1024 --force
      osprey ariel reembed --model nomic-embed-text --dimension 768 --dry-run

``osprey ariel web``
   Launch the ARIEL web interface (FastAPI server).

   Options:
      ``-p, --port`` — Port number (default: 8085)
      ``-h, --host`` — Host to bind to (default: ``127.0.0.1``)
      ``--reload`` — Enable auto-reload for development

   .. code-block:: bash

      osprey ariel web                      # Start on localhost:8085
      osprey ariel web --port 8080          # Custom port
      osprey ariel web --host 0.0.0.0       # Bind to all interfaces
      osprey ariel web --reload             # Development mode with auto-reload

``osprey ariel purge``
   Permanently delete all ARIEL data from the database.

   Options:
      ``-y, --yes`` — Skip confirmation prompt
      ``--embeddings-only`` — Only purge embedding tables, keep entries

   .. code-block:: bash

      osprey ariel purge                      # Interactive confirmation
      osprey ariel purge -y                   # Skip confirmation
      osprey ariel purge --embeddings-only    # Keep entries, clear embeddings

Examples
--------

.. code-block:: bash

   # Full setup from scratch
   osprey deploy up                           # Start PostgreSQL
   osprey ariel quickstart                    # Migrate + ingest demo data
   osprey ariel search "RF cavity"            # Search from CLI
   osprey ariel web                           # Launch web interface

   # Add semantic search
   osprey ariel enhance -m text_embedding     # Generate embeddings
   osprey ariel search "beam instability" --mode semantic

   # Live ingestion
   osprey ariel watch --interval 600          # Poll every 10 minutes

   # Model upgrade
   osprey ariel reembed --model mxbai-embed-large --dimension 1024
   osprey ariel models                        # Verify new table

============

osprey tasks
============

Browse and manage AI assistant tasks. Tasks are structured workflows for common
development activities like testing, code review, and documentation.

Syntax
------

.. code-block:: bash

   osprey tasks [COMMAND] [OPTIONS]

Commands
--------

``osprey tasks`` (no subcommand)
   Launch interactive task browser with actions like open in editor, copy to
   project, and install as Claude Code skill.

``osprey tasks list``
   Quick non-interactive list of all available tasks.

``osprey tasks copy TASK_NAME``
   Copy a task to your project's ``.ai-tasks/`` directory for use with any AI assistant.

   Options:
      ``--force`` / ``-f`` - Overwrite existing files

``osprey tasks show TASK_NAME``
   Print a task's instructions to stdout.

``osprey tasks path TASK_NAME``
   Print the path to a task's instructions file.

Examples
--------

.. code-block:: bash

   # Interactive browser (recommended)
   osprey tasks

   # List all tasks
   osprey tasks list

   # Copy task to project for any AI assistant
   osprey tasks copy pre-merge-cleanup

   # View instructions
   osprey tasks show testing-workflow

   # Get path (useful for scripting)
   osprey tasks path create-capability

Using Tasks
-----------

After copying a task to your project, reference it in your AI assistant:

.. code-block:: text

   @.ai-tasks/testing-workflow/instructions.md Help me write tests

See :doc:`../../contributing/03_ai-assisted-development` for detailed workflow guides.

osprey claude
=============

Manage Claude Code skill installations. Skills are task wrappers that enable
Claude Code to automatically discover and use Osprey workflows.

Syntax
------

.. code-block:: bash

   osprey claude [COMMAND] [OPTIONS]

Commands
--------

``osprey claude install TASK``
   Install a task as a Claude Code skill in ``.claude/skills/<task>/``.

   Skills can be installed from two sources:

   1. **Custom wrappers** - Pre-built skill files in ``integrations/claude_code/<task>/``
   2. **Auto-generated** - Generated from task frontmatter if ``skill_description`` is present

   Options:
      ``--force`` / ``-f`` - Overwrite existing installation

``osprey claude list``
   List installed and available Claude Code skills.

   Shows:

   - Installed skills in current project
   - Available skills (custom wrappers)
   - Auto-generatable skills (from task frontmatter)
   - Tasks without skill support

Examples
--------

.. code-block:: bash

   # List available and installed skills
   osprey claude list

   # Install a skill
   osprey claude install create-capability

   # Force reinstall
   osprey claude install testing-workflow --force

Output:

.. code-block:: text

   Claude Code Skills

   Installed in this project:
     ✓ create-capability

   Available to install:
     ○ migrate
     ○ testing-workflow (auto-generated)
     ○ ai-code-review (auto-generated)

   Tasks without skill support (use @-mention or add skill_description):
     - comments
     - release-workflow

Skill Auto-Generation
---------------------

Tasks with ``skill_description`` in their frontmatter can be installed as skills
without requiring custom wrappers:

.. code-block:: yaml

   ---
   workflow: my-task
   skill_description: >-
     Description of when Claude should use this skill.
     Include keywords users might say.
   ---

When installed, the skill is auto-generated from this frontmatter.

See :doc:`../../contributing/03_ai-assisted-development` for complete workflow documentation.

============

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

   Work with multiple projects simultaneously using the ``--project`` flag or ``OSPREY_PROJECT`` environment variable.

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
