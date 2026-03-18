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
   osprey deploy COMMAND     # Manage services
   osprey health             # Check system health
   osprey migrate            # Run project migrations
   osprey tasks              # Browse AI assistant tasks
   osprey claude             # Manage Claude Code integration
   osprey eject              # Copy framework components for customization
   osprey channel-finder     # Channel finder CLI
   osprey ariel              # ARIEL logbook search service
   osprey artifacts          # Artifact gallery
   osprey web                # Launch web terminal
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
   ``minimal`` (default), ``hello_world_weather``, or ``control_assistant``.

``--registry-style <style>``
   ``extend`` (default, recommended) or ``standalone``.

.. code-block:: bash

   osprey init my-agent
   osprey init weather-demo --template hello_world_weather

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

``osprey config set-models [--provider PROVIDER] [--model MODEL] [--project PATH]``
   Configure AI provider and models for all roles. Interactive if no options.

.. code-block:: bash

   osprey config show
   osprey config set-control-system epics
   osprey config set-models --provider anthropic --model claude-sonnet-4

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

osprey migrate
==============

Run project migrations for newer framework versions.

.. code-block:: bash

   osprey migrate

osprey claude
=============

Manage Claude Code integration -- skills, chat, and code resolver.

``osprey claude chat``
   Start an interactive conversation with the agent via Claude Code.

``osprey claude install TASK [--force]``
   Install a task as a Claude Code skill in ``.claude/skills/<task>/``.

``osprey claude list``
   List installed and available Claude Code skills.

.. code-block:: bash

   osprey claude chat
   osprey claude list
   osprey claude install create-capability --force

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

Launch the web terminal interface (FastAPI + PTY).

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
