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
   osprey build PROJECT      # Build project from preset or profile
   osprey config             # Manage configuration
   osprey deploy COMMAND     # Manage services
   osprey health             # Check system health
   osprey channel-finder     # Channel finder CLI
   osprey claude             # Manage Claude Code integration
   osprey eject              # Copy framework components for customization
   osprey ariel              # ARIEL logbook search service
   osprey artifacts          # Artifact gallery
   osprey web                # Launch web terminal
   osprey prompts            # Prompt artifact overrides
   osprey audit              # Audit project or profile safety
   osprey skills             # Manage bundled Osprey skills
   osprey vendor             # Manage locally bundled vendor assets

Global Options
==============

``--version``
   Show framework version and exit.

``--help``
   Show help for any command (e.g., ``osprey deploy --help``).

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

Build a facility-specific assistant from a bundled preset or a YAML profile.
See :doc:`/how-to/build-profiles`.

.. code-block:: bash

   osprey build PROJECT_NAME [PROFILE] [OPTIONS]

``--preset NAME`` — Use a bundled preset (mutually exclusive with positional
``PROFILE``). Run ``osprey build --list-presets`` to see available names.

``-O, --override PATH`` — Layer a YAML file on top of the base preset/profile
(repeatable, in order).

``--set KEY.PATH=VALUE`` — Inline scalar/list override (repeatable). RHS is
parsed as YAML so ``true``, ``[a,b]``, and bare ints/floats are typed.

``--list-presets`` — Print bundled preset names and exit.

``-o, --output-dir PATH`` — Output directory (default: current directory).

``-f, --force`` — Overwrite existing project directory.

``-s, --stream`` — Stream build step output in real time.

.. code-block:: bash

   osprey build my-agent --preset hello-world
   osprey build als-test ~/profiles/als-dev.yml --force
   osprey build edu --preset education -O overrides.yml --set model=claude-sonnet-4-6
   osprey build --list-presets

osprey deploy
=============

Manage Docker/Podman services for Osprey projects.

.. code-block:: bash

   osprey deploy ACTION [OPTIONS]

**Actions:** ``up``, ``down``, ``restart``, ``status``, ``build``, ``clean``, ``rebuild``.

- ``up`` -- Start all configured services.
- ``down`` -- Stop all services.
- ``restart`` -- Restart all services.
- ``status`` -- Show service status.
- ``build`` -- Build/prepare compose files without starting services.
- ``clean`` -- Remove containers and volumes (destructive).
- ``rebuild`` -- Clean, rebuild, and restart services.

**Options (apply to all actions):**

``-p, --project DIRECTORY`` -- Project directory (default: current directory or ``OSPREY_PROJECT``).

``-c, --config PATH`` -- Configuration file (default: ``config.yml`` in project directory).

``-d, --detached`` -- Run services in detached mode (for ``up``, ``restart``, ``rebuild``).

``--dev`` -- Copy local osprey package to containers instead of using PyPI version.

``--expose`` -- Expose services on all network interfaces (``0.0.0.0``).

.. code-block:: bash

   osprey deploy up -d
   osprey deploy status
   osprey deploy rebuild --dev
   osprey deploy down

osprey health
=============

Run comprehensive system health check.

.. code-block:: bash

   osprey health [OPTIONS]

``-p, --project DIRECTORY`` -- Project directory (default: current directory or ``OSPREY_PROJECT``).

``-v, --verbose`` -- Show detailed information about warnings and errors.

``-b, --basic`` -- Skip model completion tests (only check configuration and connectivity).

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

osprey eject
============

Copy framework services to your project for customization.

``osprey eject list``
   List all ejectable framework capabilities and services.

``osprey eject service NAME [--output PATH] [--include-tests]``
   Copy a framework service directory locally.

.. code-block:: bash

   osprey eject list
   osprey eject service channel_finder --include-tests

osprey channel-finder
=====================

Tools for building, validating, previewing, and serving control system
channel databases.

Options: ``-p, --project PATH``, ``-v, --verbose``

``osprey channel-finder build-database``
   Build a channel database from a CSV file.

``osprey channel-finder validate``
   Validate a channel database JSON file.

``osprey channel-finder preview``
   Preview a channel database with flexible display options.

``osprey channel-finder web``
   Launch the Channel Finder web interface.

.. code-block:: bash

   osprey channel-finder build-database
   osprey channel-finder validate
   osprey channel-finder preview
   osprey channel-finder web

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

Manage the OSPREY Artifact Gallery -- a local web gallery that displays
interactive plots, tables, and other outputs produced by Claude during
analysis sessions. Artifacts are written by Claude via ``save_artifact()`` in
``osprey execute`` or the ``artifact_save`` MCP tool.

``osprey artifacts web [OPTIONS]``
   Launch the Artifact Gallery web interface. Starts a FastAPI server on
   ``http://127.0.0.1:8086`` by default.

   ``-p, --port INTEGER`` — Port (default: from ``config.yml`` or ``8086``).

   ``-h, --host TEXT`` — Host to bind to (default: from ``config.yml`` or
   ``127.0.0.1``).

   ``--reload`` — Enable auto-reload for development.

.. code-block:: bash

   osprey artifacts web                    # Start on localhost:8086
   osprey artifacts web --port 9000        # Custom port
   osprey artifacts web --host 0.0.0.0     # Bind to all interfaces
   osprey artifacts web --reload           # Development mode

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

Manage prompt artifact ownership. Framework-managed prompt artifacts (agents,
rules, etc.) can be claimed per-facility for in-place editing. Claimed files
are marked user-owned in ``config.yml`` and ``.osprey-manifest.json``, and
subsequent ``osprey claude regen`` runs skip them.

All subcommands accept a common flag:

``-p, --project DIRECTORY`` — Project directory (default: current directory).

``osprey prompts list``
   List all prompt artifacts and their ownership status (framework vs.
   user-owned).

``osprey prompts claim NAME``
   Claim ownership of a framework artifact for in-place editing. If the file
   doesn't exist yet, the framework template is rendered in place at the
   canonical output path. If it already exists, it is marked user-owned.

``osprey prompts diff NAME``
   Show a unified diff between the current framework template (re-rendered)
   and your file at the canonical output path.

``osprey prompts unclaim NAME``
   Release ownership and restore framework management. The next
   ``osprey claude regen`` will overwrite the file with the framework template.

.. code-block:: bash

   osprey prompts list                           # Show all artifacts
   osprey prompts claim agents/channel-finder    # Claim for editing
   osprey prompts diff agents/channel-finder     # Compare yours vs framework
   osprey prompts unclaim rules/safety           # Restore framework management

osprey skills
=============

Manage bundled Osprey skills — Claude Code skills shipped with OSPREY that
can be installed either globally or into a specific project's
``.claude/skills/`` directory.

``osprey skills install NAME [--target PATH]``
   Install a bundled skill into ``<target>/<name>/`` (defaults to
   ``~/.claude/skills/<name>/``). If the target already exists and is
   non-empty, the prior content is renamed to
   ``<name>.bak.<YYYYMMDD-HHMMSS>/`` before the new copy is written, so a
   previous version is never lost.

   ``--target PATH`` — directory to install into. Tilde is expanded. Use a
   project-local ``.claude/skills/`` path to scope the skill to one repo
   (e.g., ``build-profile/.claude/skills/``). Omit for the global install.

   Currently supported skills:

   * ``osprey-build-interview`` — guided project-profile generation (see
     :doc:`/getting-started/osprey-build-interview`). Typically installed globally
     so it is available in any Claude Code session.
   * ``osprey-build-deploy`` — CI/CD setup, deploy-server operations, and
     release workflow for a facility profile repo. Typically installed
     project-locally (into the profile repo's ``.claude/skills/``) by the
     last phase of the ``osprey-build-interview`` skill, so ``/osprey-build-deploy``
     is available wherever the profile repo is cloned.

.. code-block:: bash

   osprey skills install osprey-build-interview
   osprey skills install osprey-build-deploy --target build-profile/.claude/skills/

osprey vendor
=============

Manage locally bundled vendor assets (JS/CSS/fonts) for firewalled
deployments. By default OSPREY interfaces load third-party libraries directly
from CDN; set ``OSPREY_OFFLINE=1`` (or ``offline: true`` in ``config.yml``) to
switch the interfaces over to local bundles.

``osprey vendor fetch [OPTIONS]``
   Download all vendor assets declared in the manifest into
   ``static/vendor/``. Run once on firewalled deployments before starting
   ``osprey web`` with ``OSPREY_OFFLINE=1``. In default CDN mode this command
   is optional.

   ``-q, --quiet`` — Suppress per-file output.

   ``-k, --insecure`` — Skip TLS cert verification. Every asset is still
   checked against its manifest SHA256, so this is safe behind corporate
   proxies (e.g. Squid) that intercept TLS. Also enabled via
   ``OSPREY_VENDOR_INSECURE=1``.

``osprey vendor verify``
   Verify all vendor assets exist on disk with correct SHA256 checksums.

.. code-block:: bash

   osprey vendor fetch                    # Download all assets
   osprey vendor fetch --insecure         # Behind a TLS-intercepting proxy
   osprey vendor verify                   # Check checksums

Environment Variables
=====================

.. code-block:: bash

   OSPREY_PROJECT=/path/to/project   # Default project directory
   ANTHROPIC_API_KEY=sk-...          # Or OPENAI_API_KEY, GOOGLE_API_KEY, etc.

``OSPREY_PROJECT`` sets a default project directory for all commands. Priority:
``--project`` flag > ``OSPREY_PROJECT`` > current directory.
