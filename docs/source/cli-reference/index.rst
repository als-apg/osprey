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
   osprey profile            # Author, validate, and inspect build profiles
   osprey build PROJECT      # Build a project from a build profile
   osprey config             # Manage configuration
   osprey deploy COMMAND     # Manage services
   osprey health             # Check system health
   osprey channel-finder     # Channel finder CLI
   osprey claude             # Manage Osprey agent integration
   osprey eject              # Copy framework components for customization
   osprey ariel              # ARIEL logbook search service
   osprey artifacts          # Artifact gallery
   osprey web                # Launch web terminal
   osprey theme-lab          # Design and preview a theme in the browser
   osprey scaffold           # Build artifact overrides
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
   Switch connector: ``mock`` or ``epics``. (Other control systems are
   reachable through custom connector packages — see
   :doc:`/how-to/add-connector`.)

``osprey config set-epics-gateway [--facility als|aps|custom] [--address] [--port]``
   Configure EPICS gateway using facility presets or custom values.

.. code-block:: bash

   osprey config show
   osprey config set-control-system epics

osprey profile
==============

Author, validate, and inspect build profiles. A profile directory is the
durable, facility-owned input to ``osprey build`` — see
:doc:`/how-to/build-profiles`.

.. code-block:: bash

   osprey profile new TARGET_DIR --preset NAME [OPTIONS]
   osprey profile validate TARGET
   osprey profile presets

``osprey profile new TARGET_DIR --preset NAME``
   Create a **facility repository** from a bundled preset. ``TARGET_DIR`` is
   the repository this facility's deployment lives in, and the command writes
   the whole thing:

   .. code-block:: text

      TARGET_DIR/
        profile/       the editable source the facility owns
        build/         empty; where `osprey build` renders projects
        ci-extra.yml   the facility's own CI jobs; never regenerated
        .gitignore     keeps build/ and the profile's secrets out of git

   ``profile/`` holds a standalone ``profile.yml`` (the preset's full
   configuration written out explicitly, no ``extends:``), the preset's
   ``data/`` tree copied verbatim, an ``.env.example`` listing every variable
   the agent reads, an ``.env`` seeded from your shell (only when it held keys
   for a provider this profile references), its own ``.gitignore``, and a
   ``README.md``. Directories for your own artifacts (``rules/``, ``skills/``,
   …) are not created up front — make the ones you need.

   ``git init`` runs at the repository root and nothing is committed. The CI
   pipeline is emitted too, as soon as there is anything to render it from: a
   profile whose ``deploy:`` block is still the commented stub gets the rest of
   the repository, and ``osprey deploy scaffold`` adds the pipeline once the
   block is filled in. Refuses to write into an existing directory unless
   ``--force`` is given.

   ``-O, --override PATH`` — Layer a YAML file on top of the preset before
   writing (repeatable, in order).

   ``--set KEY.PATH=VALUE`` — Inline override baked into the written profile
   (repeatable). RHS is parsed as YAML. Wins over ``-O`` at the same key.

   ``--force`` — Replace an existing repository's ``profile/`` directory,
   deleting its current contents including any edits you made there, and
   overwrite hand-edited deployment files. A target that is neither a facility
   repository (no ``profile/`` directory) nor empty is refused. ``ci-extra.yml``
   and the repository's ``.gitignore`` are never touched. Nothing is deleted
   until the replacement profile has fully rendered, so a failed run leaves the
   old directory untouched.

``osprey profile validate TARGET``
   Check a profile without building anything. ``TARGET`` is a profile
   directory (its ``profile.yml`` is used) or a path to a profile file.
   Resolves ``extends:`` chains and reports every problem found — convention
   directories, the ``data:`` tree, service templates, lifecycle steps, env
   vars. Exits 0 when valid, 2 with the accumulated errors when not.

``osprey profile presets``
   List bundled preset names, one per line. Every name printed is usable as
   ``--preset NAME`` for ``osprey profile new`` and ``osprey build``.

.. code-block:: bash

   osprey profile presets
   osprey profile new my-facility --preset control-assistant --set model=opus
   cd my-facility
   osprey profile validate profile/
   osprey build my-agent profile/            # renders into build/my-agent/

osprey build
============

Build a facility-specific assistant from a build profile. Every build reads a
profile — there is no build straight out of a bundled preset. See
:doc:`/how-to/build-profiles`.

.. code-block:: bash

   osprey build PROJECT_NAME [PROFILE] [OPTIONS]

``--preset NAME`` — Materialize ``<PROJECT_NAME>-profile/`` from a bundled
preset and build from it (mutually exclusive with positional ``PROFILE``). Only
the *first* such build materializes; every later one reuses that directory as it
stands. Run ``osprey build --list-presets`` to see available names.

``-O, --override PATH`` — Layer a YAML file on top of the profile (repeatable,
in order). Written into the profile when it already exists.

``--set KEY.PATH=VALUE`` — Inline scalar/list override (repeatable). RHS is
parsed as YAML so ``true``, ``[a,b]``, and bare ints/floats are typed. Written
into the profile when it already exists, replacing the value at the dotted key
path.

``--list-presets`` — Print bundled preset names and exit.

``-o, --output-dir PATH`` — Render the project under this directory, overriding
the default. A profile nested in a facility repository (``<repo>/profile/``)
renders into ``<repo>/build/<PROJECT_NAME>/`` whichever directory the command is
run from; anything else renders under the current directory.

``-f, --force`` — Re-render an existing project directory in place; ``.env``,
``_agent_data/``, and ``.git`` are preserved. Never touches the profile —
replace one with ``osprey profile new --force``.

``--tier [1|3]`` — Channel-database tier. Selects which
``data/channel_databases/tiers/tier{N}/`` database the rendered config points
at, overriding the paradigm-derived default. Written into the profile like
``--set``.

``-s, --stream`` — Stream build step output in real time.

``--skip-lifecycle`` — Skip the profile's ``pre_build``, ``post_build``, and
``validate`` steps.

``--skip-deps`` — Skip venv creation and dependency installation (CI mode).

``--runtime-root PATH`` — Override ``project_root`` in the rendered config, for
container builds where the build path differs from the runtime path.

.. code-block:: bash

   osprey build my-agent --preset hello-world
   osprey build my-facility profile/          # from a facility repo → build/my-facility/
   osprey build als-test ~/profiles/als-dev.yml --force
   osprey build edu --preset education -O overrides.yml --set model=claude-sonnet-4-6
   osprey build --list-presets

osprey deploy
=============

Manage Docker/Podman services for Osprey projects.

.. code-block:: bash

   osprey deploy VERB [OPTIONS]

Each verb declares its own options. A flag a verb does not take is a **parse
error** (exit 2), not a silent no-op — ``osprey deploy status --dev`` fails
rather than ignoring the flag. Run ``osprey deploy VERB --help`` for one verb's
exact set.

Every verb acts on one project: run it from the project directory, pass
``-p/--project``, or set ``OSPREY_PROJECT``. The exception is ``scaffold``,
which emits a facility repository's deployment files and therefore acts on the
repository rather than on a project built from it.

**Service verbs.** All of these take ``-p/--project`` and ``-c/--config``;
the third column lists what each one takes on top of those.

.. list-table::
   :header-rows: 1
   :widths: 16 46 38

   * - Verb
     - What it does
     - Also accepts
   * - ``up``
     - Start all configured services.
     - ``-d/--detached``, ``--dev``, ``--expose``
   * - ``down``
     - Stop all services.
     - ``--dev``
   * - ``restart``
     - Restart all services.
     - ``-d/--detached``, ``--expose``
   * - ``status``
     - Show service status.
     - —
   * - ``build``
     - Build/prepare compose files without starting services.
     - ``--dev``, ``--expose``
   * - ``clean``
     - Remove containers and volumes (destructive).
     - ``--dev``, ``--expose``
   * - ``rebuild``
     - Clean, rebuild, and restart services.
     - ``-d/--detached``, ``--dev``, ``--expose``

``-p, --project DIRECTORY`` -- Project directory (default: current directory or ``OSPREY_PROJECT``).

``-c, --config PATH`` -- Configuration file (default: ``config.yml`` in project directory).

``-d, --detached`` -- Run services in detached mode.

``--dev`` -- Copy local osprey package to containers instead of using PyPI version.

``--expose`` -- Expose services on all network interfaces (``0.0.0.0``). Only
use this with authentication configured.

**Web-terminal workspace verbs.** These also take ``-p/--project`` and
``-c/--config``.

.. list-table::
   :header-rows: 1
   :widths: 16 46 38

   * - Verb
     - What it does
     - Also accepts
   * - ``decommission USER``
     - Remove a single user's web-terminal workspace.
     - ``--archive``, ``--purge``, ``-y/--yes``
   * - ``prune``
     - Remove workspaces for users no longer in the user index.
     - ``--archive``, ``--purge``, ``-y/--yes``, ``--dry-run``
   * - ``nuke``
     - Tear down the whole multi-user web-terminal stack. Destructive: it
       removes every user's workspace, not just the stale ones.
     - ``-y/--yes``
   * - ``seed [USER]``
     - (Re)seed workspaces from the user index; ``USER`` targets one user, omit
       to reseed all.
     - —
   * - ``passwd USER``
     - Change one web-terminal user's login password (password authentication
       only). Prompts without echoing, and ends that user's sessions.
     - —

``--archive`` -- Archive a user's workspace before removing it (mutually exclusive with ``--purge``).

``--purge`` -- Permanently delete a user's workspace without archiving it (mutually exclusive with ``--archive``).

``-y, --yes`` -- Assume yes to confirmation prompts.

``--dry-run`` -- Show what would happen without making changes.

**Deployment-file verbs.**

``osprey deploy scaffold [--repo DIRECTORY] [--force]``
   Emit a facility repository's deployment files from its profile's ``deploy:``
   block: the CI pipeline at the repository root, and the post-deploy health
   check inside the profile's ``project/`` mirror, which every build copies to
   ``scripts/verify.sh``. Run it from anywhere inside the repository and it
   finds the root on its own; ``--repo`` names one explicitly. Re-running is
   safe — a file whose content already matches is left untouched, stamp
   included, so an OSPREY upgrade alone produces no diff. A file the scaffolder
   did not write is reported and left alone unless ``--force`` is given.

``osprey deploy render-env-production [OPTIONS]``
   Render ``.env.production`` — the env file every per-user web-terminal
   container runs with — from the deploy config and one secrets file. It
   applies no rule of its own, so a file rendered here and one generated by
   ``osprey deploy up`` cannot disagree. Secret values come only from the
   secrets file, never from the surrounding environment.

   ``--env-file PATH`` — Secrets file to render from (default: ``.env`` in the
   project directory).

   ``-o, --output PATH`` — Write the result here, at mode ``0600``, instead of
   to stdout. Unlike a deploy, which never overwrites an existing
   ``.env.production``, an explicit ``--output`` is taken as an instruction and
   replaces what is there. In CI, pass it: without ``--output`` the assembled
   secrets go to the job log.

.. code-block:: bash

   osprey deploy up -d
   osprey deploy status
   osprey deploy rebuild --dev
   osprey deploy down
   osprey deploy decommission alice --archive
   osprey deploy prune --dry-run
   osprey deploy nuke --yes
   osprey deploy scaffold
   osprey deploy render-env-production --output .env.production

See :doc:`/how-to/deploy-a-facility` for the walkthrough that uses these
verbs end to end.

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

Manage Osprey agent integration — regenerate artifacts, launch chat, and check
status.

``osprey claude chat [OPTIONS]``
   Regenerate artifacts from ``config.yml``, launch companion servers, and
   start the Osprey agent in the terminal. See :doc:`/how-to/use-cli-chat`.

   ``-p, --project DIRECTORY`` — Project directory (default: current directory).

   ``--resume SESSION_ID`` — Resume a previous session.

   ``--print`` — Non-interactive pipe-friendly mode.

   ``--effort [low|medium|high|max]`` — Set effort level.

``osprey claude regen [OPTIONS]``
   Re-render all Osprey agent integration files (``.mcp.json``,
   ``.claude/settings.json``, ``CLAUDE.md``, agents) from ``config.yml``.
   Existing files are backed up to ``_agent_data/backup/``.

   ``-p, --project DIRECTORY`` — Project directory (default: current directory).

   ``--dry-run`` — Show what would change without writing files.

   ``--runtime-root PATH`` — Rewrite ``project_root`` in ``config.yml`` to
   PATH (comment-preserving) and re-render artifacts against it. Use after
   copying a built project into a container image; see
   :doc:`/how-to/containerize-project`.

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

``osprey channel-finder generate [--output-dir DIR] [--source PATH] [--format in_context|hierarchical|middle_layer|all] [--tier 1|3|none] [--validate]``
   Generate channel databases from a hierarchical template. Produces one
   or more pipeline formats (default: all three) with optional tier filtering.

``osprey channel-finder benchmark --model PROVIDER/WIRE_ID [--queries SPEC] [--runs-per-query N] [--concurrency N] [--output-dir DIR] [--queries-path PATH] [-v]``
   Run the benchmark harness against a channel-finder pipeline using a
   LiteLLM-form model id (e.g. ``anthropic/claude-haiku-4-5``). Saves per-run
   JSON results for accuracy/cost analysis.

``osprey channel-finder web``
   Launch the Channel Finder web interface.

.. code-block:: bash

   osprey channel-finder build-database
   osprey channel-finder validate
   osprey channel-finder preview
   osprey channel-finder generate --format hierarchical
   osprey channel-finder benchmark --model anthropic/claude-haiku-4-5
   osprey channel-finder web

osprey ariel
============

Manage the ARIEL logbook search service.

``quickstart [--source PATH]`` -- Full setup: migrate and ingest demo data.

``status [--json]`` -- Show service status.

``migrate`` -- Create or update database tables.

``sync [--limit N]`` -- Idempotent migrate + incremental ingest + enhance.
Safe to run on every build; on a fresh database, runs a full ingest.

``ingest --source PATH [--adapter TYPE] [--since DATE] [--limit N] [--dry-run]``
   Ingest logbook entries from file or URL.

``watch [--source] [--once] [--interval N] [--dry-run]`` -- Poll for new entries.

``enhance [--module NAME] [--force] [--limit N]`` -- Run enhancement modules.

``models`` -- List embedding models and tables.

``search QUERY [--mode keyword|semantic] [--limit N] [--json]``
   Execute a search query (default mode: ``keyword``).

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
interactive plots, tables, and other outputs produced by the Osprey agent during
analysis sessions. Artifacts are written by the Osprey agent via ``save_artifact()`` in
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

Launch the Web Terminal interface. See :doc:`/how-to/web-terminal/operate`.

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

osprey theme-lab
================

Design a theme in the browser. Starts a local server for OSPREY's design
system and opens the Theme Lab, where you pick an accent color and see it
previewed live on dark and light mock-ups of the web terminal, with contrast
badges that update as you go. Copying the export block gives you a
ready-to-paste description of the theme to request; the lab itself does not
write theme files. See :doc:`/how-to/web-terminal/theming`.

``osprey theme-lab [OPTIONS]``
   Serve the Theme Lab and open it. The URL is printed as well, so the page can
   be opened by hand if no browser appears.

   ``-p, --port INTEGER`` — Port to serve on (default: an unused port chosen
   automatically).

   ``--no-browser`` — Do not open a browser window; print the URL only.

.. code-block:: bash

   osprey theme-lab
   osprey theme-lab --port 9000
   osprey theme-lab --no-browser

osprey audit
============

Audit a build profile or project directory for safety risks. Uses an AI
reviewer to analyze permissions, hooks, MCP server configs, convention
directories, and lifecycle scripts.

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

osprey scaffold
===============

Manage build artifact ownership. Framework-managed build artifacts (agents,
rules, etc.) can be claimed per-facility for in-place editing. A claim moves
the artifact into the profile the project was built from; the next build copies
it back and registers it as user-owned, so ``osprey claude regen`` skips it.

All subcommands accept a common flag:

``-p, --project DIRECTORY`` — Project directory (default: current directory).

``osprey scaffold list``
   List all build artifacts and their ownership status (framework vs.
   user-owned).

``osprey scaffold claim NAME``
   Move an artifact into the profile this project was built from, into the
   convention directory for its kind (``rules/safety.md``,
   ``skills/orbit-check/``, ``services/postgresql/``, ``hooks/my-guard``). A
   file moves as a file; skills and services move as whole directories. The
   project copy is *moved*, not copied — it lives in one place until the next
   ``osprey build ... --force`` deploys it again.

   Refused, with the reason: a project with no resolvable profile (nothing
   would keep the edit); a **generated** artifact rather than an authored one —
   ``CLAUDE.md``, ``.claude/settings.json``, ``.mcp.json``,
   ``hook_config.json`` — where the message names the config key that does
   control it; and a profile slot that is already occupied. See
   :ref:`profile-claim`.

``osprey scaffold diff NAME``
   Show a unified diff between the current framework template (re-rendered)
   and your file at the canonical output path. For a claimed service
   directory, diffs every file in the directory against the packaged
   template.

``osprey scaffold unclaim NAME``
   Release ownership and restore framework management. The next
   ``osprey claude regen`` will overwrite the file with the framework template.
   Ownership a build derived from the profile is re-registered by the next
   build, so this holds only until then — give the artifact up for good by
   deleting it from the profile's convention directory.

``osprey scaffold web-terminals lint [-p PATH]``
   Validate a project's ``modules.web_terminals`` stanza (port-family
   allocation, reserved service names, duplicate users, persona references).
   Exits non-zero on error-severity findings; warnings do not fail the check,
   so it is safe to wire into a CI gate.

``osprey scaffold web-terminals render [-p PATH] -o DIRECTORY``
   Render the project's multi-user deployment artifacts (docker-compose
   overlay, nginx routing fragment, static landing page) into ``-o/--output``.
   Lints first by default and aborts on errors; ``--no-lint`` skips the
   pre-check.

   Both verbs read the stanza from the project's ``config.yml``, selected with
   ``-p/--project`` (default: the current directory).

.. code-block:: bash

   osprey scaffold list                           # Show all artifacts
   osprey scaffold claim agents/channel-finder    # Claim for editing
   osprey scaffold claim services/postgresql      # Freeze a service template
   osprey scaffold diff agents/channel-finder     # Compare yours vs framework
   osprey scaffold unclaim rules/safety           # Restore framework management
   osprey scaffold web-terminals lint             # lint this project's stanza
   osprey scaffold web-terminals render -o deploy/

osprey skills
=============

Manage bundled Osprey skills — agent skills shipped with OSPREY that
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
   (e.g., a facility repository's ``.claude/skills/``). Omit for the global
   install.

   Currently supported skills:

   * ``osprey-build-interview`` — guided facility-repository generation (see
     :doc:`/getting-started/osprey-build-interview`). Typically installed globally
     so it is available in any Osprey agent session.
   * ``osprey-deploy-ops`` — the operate-time runbook: emitting the CI and
     health-check files from the profile's ``deploy:`` block, bringing the stack
     up on the deploy host, and triaging it when a service is down. Typically
     installed project-locally (into the facility repository's
     ``.claude/skills/``) so it travels with the repository.
   * ``creating-an-osprey-panel`` — author a themed, token-only web-terminal
     panel.
   * ``osprey-contribute`` — walks a contributor through the GitHub Flow
     journey from a working-tree change to a merged PR on ``main`` (branching,
     atomic commits, push, PR, rebase, merge).
   * ``osprey-pre-commit`` — runs the quick / ci / premerge check scripts at
     the right gate before committing, pushing, or opening a PR.
   * ``osprey-release`` — cuts a CalVer release: opens the version-bump PR,
     tags the merge commit, and verifies the automated PyPI publish.
   * ``osprey-design-philosophy`` — OSPREY's design and architecture principles
     for designing, adding, or reviewing a feature. Useful for framework
     contributors; install globally to have it available when working on
     ``src/osprey`` in any session.

.. code-block:: bash

   osprey skills install osprey-build-interview
   osprey skills install osprey-deploy-ops --target .claude/skills/

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
