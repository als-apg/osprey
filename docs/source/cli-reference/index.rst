=============
CLI Reference
=============

Complete reference for all Osprey Framework CLI commands.

**Prerequisites:** Framework installed (``uv sync``)

Overview
========

All commands are accessed through the ``osprey`` command. Running ``osprey``
without arguments launches an interactive TUI menu.

.. code-block:: bash

   osprey                    # Launch interactive menu
   osprey --version          # Show framework version
   osprey init PROJECT       # Create new project
   osprey config             # Manage configuration
   osprey build              # Build project from profile
   osprey deploy COMMAND     # Manage services
   osprey health             # Check system health
   osprey tasks              # Browse AI assistant tasks
   osprey claude             # Manage Claude Code integration
   osprey web                # Launch web terminal
   osprey audit              # Audit project or profile safety
   osprey eject              # Copy framework components for customization
   osprey channel-finder     # Channel finder CLI
   osprey ariel              # ARIEL logbook search service
   osprey artifacts          # Artifact gallery
   osprey prompts            # Prompt artifact overrides

Global Options
==============

``--version``
   Show framework version and exit.

``--help``
   Show help for any command (e.g., ``osprey deploy --help``).

osprey init
===========

Create a new project from a template.

.. code-block:: bash

   osprey init [OPTIONS] PROJECT_NAME

``--template <name>``
   ``hello_world``, ``control_assistant`` (default), or ``lattice_design``.

``--registry-style <style>``
   ``extend`` (default, recommended) or ``standalone``.

.. code-block:: bash

   osprey init my-agent
   osprey init my-first-agent --template hello_world

osprey config
=============

Manage project configuration. Interactive menu if no subcommand is given.

``osprey config show [--project PATH] [--format yaml|json]``
   Display current project configuration.

``osprey config export [--output PATH] [--format yaml|json]``
   Export framework default configuration template.

``osprey config set-control-system SYSTEM_TYPE [--project PATH]``
   Switch connector: ``mock``, ``epics``, ``tango``, ``labview``.

``osprey config set-epics-gateway [--facility als|aps|custom] [--address] [--port]``
   Configure EPICS gateway using facility presets or custom values.

.. code-block:: bash

   osprey config show
   osprey config set-control-system epics

osprey build
============

Build a facility-specific assistant from a profile. See
:doc:`/how-to/build-profiles`.

.. code-block:: bash

   osprey build PROJECT_NAME PROFILE [OPTIONS]

``-o, --output-dir PATH`` — Output directory (default: current directory).

``-f, --force`` — Overwrite existing project directory.

``-s, --stream`` — Stream build step output in real time.

.. code-block:: bash

   osprey build als-test ~/profiles/als-dev.yml
   osprey build my-assistant profile.yml --force -o /tmp

osprey deploy
=============

Manage containerized services. All subcommands accept ``--project PATH``.

``up [--detached] [--dev]``
   Start services.

``down``
   Stop all running services.

``restart``
   Restart all services.

``status``
   Show status of deployed services.

``clean``
   Stop services and remove containers and volumes.

``rebuild [--detached] [--dev]``
   Rebuild containers from scratch.

.. code-block:: bash

   osprey deploy up --detached
   osprey deploy status
   osprey deploy rebuild --dev
   osprey deploy down

osprey health
=============

Run comprehensive system health check.

.. code-block:: bash

   osprey health [--project PATH]

osprey claude
=============

Manage Claude Code integration — regenerate artifacts, launch chat, and check
status.

``osprey claude chat [OPTIONS]``
   Regenerate artifacts from ``config.yml``, launch companion servers, and
   start Claude Code in the terminal. See :doc:`/how-to/use-cli-chat`.

   ``-p, --project DIRECTORY`` — Project directory (default: current directory).

   ``--resume SESSION_ID`` — Resume a previous session.

   ``--print`` — Non-interactive pipe-friendly mode.

   ``--effort [low|medium|high|max]`` — Set effort level.

``osprey claude regen [OPTIONS]``
   Re-render all Claude Code integration files (``.mcp.json``,
   ``.claude/settings.json``, ``CLAUDE.md``, agents) from ``config.yml``.
   Existing files are backed up to ``_agent_data/backup/``.

   ``-p, --project DIRECTORY`` — Project directory (default: current directory).

   ``--dry-run`` — Show what would change without writing files.

``osprey claude status [OPTIONS]``
   Display provider configuration, model tier mappings, per-agent model
   assignments, and artifact sync status.

   ``-p, --project DIRECTORY`` — Project directory (default: current directory).

.. code-block:: bash

   osprey claude chat
   osprey claude chat --resume abc123
   osprey claude regen --dry-run
   osprey claude status

osprey tasks
============

Browse and manage AI assistant tasks (structured development workflows).

``osprey tasks``
   Launch interactive task browser.

``osprey tasks list``
   Non-interactive list of all available tasks.

``osprey tasks copy TASK_NAME [--force]``
   Copy a task to ``.ai-tasks/``.

``osprey tasks show TASK_NAME``
   Print task instructions to stdout.

``osprey tasks path TASK_NAME``
   Print path to task instructions file.

.. code-block:: bash

   osprey tasks list
   osprey tasks copy pre-merge-cleanup
   osprey tasks show testing-workflow

osprey eject
============

Copy framework capabilities or services to your project for customization.

``osprey eject list``
   List all ejectable components.

``osprey eject capability NAME [--output PATH] [--include-tests]``
   Copy a framework capability locally.

``osprey eject service NAME [--output PATH] [--include-tests]``
   Copy a framework service directory locally.

.. code-block:: bash

   osprey eject list
   osprey eject capability channel_finding
   osprey eject service channel_finder --include-tests

osprey channel-finder
=====================

Natural language channel search with REPL, queries, and benchmarking.

Options: ``--project PATH``, ``--verbose``

``osprey channel-finder``
   Launch interactive REPL.

``osprey channel-finder query "QUERY_TEXT"``
   Execute a single query.

``osprey channel-finder benchmark [--queries] [--model] [--dataset]``
   Run benchmarks against datasets.

``osprey channel-finder build-database [--csv PATH] [--output PATH] [--use-llm]``
   Build channel database from CSV.

``osprey channel-finder validate [--database PATH] [--verbose] [--pipeline]``
   Validate a channel database JSON file.

``osprey channel-finder preview [--depth N] [--max-items N] [--sections] [--full]``
   Preview a channel database with tree visualization.

.. code-block:: bash

   osprey channel-finder
   osprey channel-finder query "find beam position monitors"
   osprey channel-finder benchmark --queries 0:10
   osprey channel-finder preview --sections tree,stats

osprey ariel
============

Manage the ARIEL logbook search service.

``quickstart [--source PATH]`` -- Full setup: migrate and ingest demo data.

``status [--json]`` -- Show service status.

``migrate`` -- Create or update database tables.

``ingest --source PATH [--adapter TYPE] [--since DATE] [--limit N] [--dry-run]``
   Ingest logbook entries from file or URL.

``watch [--source] [--once] [--interval N] [--dry-run]`` -- Poll for new entries.

``enhance [--module NAME] [--force] [--limit N]`` -- Run enhancement modules.

``models`` -- List embedding models and tables.

``search QUERY [--mode auto|keyword|semantic|rag] [--limit N] [--json]``
   Execute a search query.

``reembed --model NAME --dimension N [--batch-size N] [--force]``
   Re-embed entries with a different model.

``web [--port N] [--host ADDR] [--reload]`` -- Launch web interface.

``purge [--yes] [--embeddings-only]`` -- Delete all ARIEL data.

.. code-block:: bash

   osprey ariel quickstart
   osprey ariel search "RF cavity fault"
   osprey ariel web --port 8080

osprey artifacts
================

Browse and organize generated outputs in the artifact gallery.

osprey web
==========

Launch the Web Terminal interface. See :doc:`/how-to/use-web-terminal`.

``osprey web [OPTIONS]``
   Start the web terminal server (default: ``http://127.0.0.1:8087``).

   ``-p, --port INTEGER`` — Port (default: from config or 8087).

   ``--host TEXT`` — Host to bind to (default: ``127.0.0.1``).

   ``--shell TEXT`` — Shell command to run (default: ``claude``).

   ``--project DIRECTORY`` — Project directory (default: current directory).

   ``--detach`` — Run in background (PID written to ``.osprey-web.pid``).

   ``--reload`` — Auto-reload for development.

``osprey web stop``
   Stop a background web terminal server.

.. code-block:: bash

   osprey web
   osprey web --port 9000 --host 0.0.0.0
   osprey web --detach
   osprey web stop

osprey audit
============

Audit a build profile or project directory for safety risks. Uses an AI
reviewer to analyze permissions, hooks, MCP server configs, overlay files,
and lifecycle scripts.

.. code-block:: bash

   osprey audit TARGET [OPTIONS]

``--build`` — Build a profile in a temp directory, then audit the result.

``--model TEXT`` — Model for the reviewer agent.

``--budget FLOAT`` — Maximum budget in USD.

``-v, --verbose`` — Show verbose output.

``--json`` — Output as JSON.

.. code-block:: bash

   osprey audit my-project/
   osprey audit profile.yml --build
   osprey audit project/ --json

osprey prompts
==============

Manage prompt artifact overrides for customizing framework prompt templates.

Environment Variables
=====================

.. code-block:: bash

   OSPREY_PROJECT=/path/to/project   # Default project directory
   ANTHROPIC_API_KEY=sk-...          # Or OPENAI_API_KEY, GOOGLE_API_KEY, etc.

``OSPREY_PROJECT`` sets a default project directory for all commands. Priority:
``--project`` flag > ``OSPREY_PROJECT`` > current directory.
