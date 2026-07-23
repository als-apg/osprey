.. _how-to-build-profiles:

==============
Build Profiles
==============

Assemble facility-specific assistants from OSPREY templates using declarative YAML profiles.
Build profiles separate **facility-specific inputs** (channel data, safety limits,
custom MCP servers) from **what OSPREY provides** (agents, rules, hooks, safety infrastructure).

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Writing build profile YAML files for your facility
   - Overlaying facility data onto OSPREY templates
   - Injecting custom MCP servers into the built project
   - Using config overrides, lifecycle commands, and environment templates
   - Structuring a facility profiles repository

   **Prerequisites:** A working OSPREY installation (``uv sync``).

   **Time:** 15--30 minutes for a basic profile; varies for custom MCP servers.

Overview
========

The ``osprey build`` command takes a YAML profile and produces a standalone Osprey agent
project. The profile declares:

- **Data bundle** to start from (``control_assistant``, ``hello_world``, or ``ariel_standalone``)
- **Config overrides** for the generated ``config.yml`` (dot-notation)
- **File overlays** that copy facility data into the project
- **MCP server definitions** to inject custom tools
- **Lifecycle commands** to run before/after the build
- **Environment templates** for required variables and defaults
- **Dependencies** to append to ``requirements.txt``

.. mermaid::

   flowchart LR
      P[profile.yml] --> B[osprey build]
      T[Base Template] --> B
      D[Facility Data] --> B
      S[Custom MCP Servers] --> B
      B --> O[Standalone Project]

The built project is **wipe-and-rebuild safe** — regenerating from the same profile
produces the same output, and user-owned files (like ``facility.md``) are tracked
separately.


Preset → Profile → Project
==========================

OSPREY's build inputs form a three-layer hierarchy:

- **Preset** — bundled upstream, lives in ``src/osprey/profiles/presets/``.
  Edited only by PR'ing OSPREY itself. A preset may itself extend another
  preset (e.g. ``operator`` extends ``control-assistant``).
  Examples: ``hello-world``, ``control-assistant``, ``ariel-standalone``,
  ``channel-finder-standalone``.
- **Profile** — user-owned, lives in *your* repo as a directory. Has
  ``extends: <preset-name>`` (or a path) and overrides only what your
  facility needs. This is your durable source-of-truth for customizations.
- **Project** — the rendered output of ``osprey build``. Derived; regenerable;
  treat it as a build artifact and avoid editing it in place.

You can drive ``osprey build`` in three modes:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Mode
     - Command
     - When to use
   * - Quick start
     - ``osprey build my-project --preset X``
     - Trying a preset; no customizations needed.
   * - Scaffold profile
     - ``osprey build --emit-profile my-profile --preset X``
     - Starting facility-specific customization. Writes an editable
       profile directory and exits — no project rendered yet.
   * - Build from profile
     - ``osprey build my-project my-profile/profile.yml``
     - Rendering a project from your profile (the everyday command after
       the profile exists).

The scaffold mode writes:

.. code-block:: text

   my-profile/
     profile.yml          # extends: <preset>, with override sections (commented)
     overlays/
       rules/   .gitkeep  # drop facility-specific rule .md files here
       skills/  .gitkeep  # drop custom skill directories here
       agents/  .gitkeep  # drop custom subagent .md files here
     README.md            # explains the layout

The seed ``profile.yml`` lists every supported override section in commented
form (``skills:``, ``rules:``, ``agents:``, ``config:``, ``env:``,
``overlay:``) — uncomment what you need.

Inheriting from a preset
------------------------

``extends:`` accepts either a bundled preset name or a filesystem path:

.. code-block:: yaml

   # By bundled preset name (recommended for new profiles).
   extends: control-assistant

   # By relative path (e.g. multi-profile facility repos with a shared base).
   extends: ./als-base.yml

Path-shaped values containing ``.yml`` always resolve as filesystem paths,
so existing multi-file layouts (like the ALS profiles repo) keep working
unchanged. A preset-name probe is tried first; if it misses, path
resolution runs.


.. _profile-exclude:

Excluding inherited entries
---------------------------

``extends:`` union-merges your string lists onto the base's, so a child can only
*add* to the sets it inherits — never remove an entry the base contributed. The
``exclude:`` key subtracts entries from those inherited lists:

.. code-block:: yaml

   extends: control-assistant

   exclude:
     skills:
       - writing-bluesky-plans

``exclude:`` maps a field name to a list of entries to remove. It accepts only
the string-list fields a profile inherits: ``skills``, ``rules``, ``hooks``,
``agents``, ``output_styles``, ``web_panels``, and ``dependencies``. Naming any
other field (e.g. the ``config`` mapping) is a load-time error.

**Layering.** ``exclude:`` is applied after each ``extends`` merge, against that
layer's merged result. Two consequences follow, and the second is easy to trip
over:

- A *deeper* ``extends`` layer that re-declares an excluded entry wins. If a
  base lists ``[a, b, c]``, a middle profile excludes ``b``, and a profile that
  extends the middle profile re-adds ``b``, then ``b`` survives — the re-add
  merges in after the exclusion ran.
- An **override file** (``-O``) or a **``--set``** value cannot re-add an
  excluded entry. Both merge into the top profile layer *before* ``extends`` is
  resolved, so the same exclusion strips them out again. To restore an excluded
  entry you must edit the profile that declares the ``exclude:`` (or a layer
  deeper than it) — you cannot layer it back on from the command line.

Excluding an entry the base never declared is a silent no-op. Declaring
``exclude:`` in a profile that has no ``extends:`` only sees the profile's own
lists — there is nothing inherited to remove — so it does nothing useful.

**Worked example: a scan-free tier.** A facility profile can build a
restricted tier on top of ``control-assistant`` by excluding the scan-related
skills:

.. code-block:: yaml

   name: Control Assistant (No Scans)
   extends: control-assistant

   exclude:
     skills:
       - writing-bluesky-plans
       - operating-bluesky-scans

   config:
     claude_code.servers.bluesky.enabled: false

The ``exclude:`` drops the scan skills from the inherited skill set, so this
tier never enters the scan-plan workflow; the ``config:`` override keeps the
bluesky MCP server off as well. Because a user building from this profile
cannot re-add a skill with ``--set skills=[writing-bluesky-plans]`` (that
merges pre-exclusion and is stripped again), the restriction holds unless the
profile itself is edited.

.. note::

   ``exclude:`` carves a tier by *removing* capability. When the boundary you
   want is "may not write," prefer flipping the enforcement switch instead —
   the bundled ``multi-user-demo-readonly`` preset differs from its
   write-capable sibling only on ``control_system.writes_enabled``, leaving
   the tool surface identical (see :doc:`web-terminal/multi-user-demo`).

To keep the scan server **on** while hiding an individual plan, set
``bluesky.excluded_plans`` on the deploying project's profile:

.. code-block:: yaml

   bluesky:
     excluded_plans: [orm]

The named plan is then invisible to the agent and non-runnable. The deploy render
carries it to the bridge as ``BLUESKY_EXCLUDED_PLANS`` (the config key alone is a
dev-only convenience).


Quick Start
===========

Create a minimal profile and build:

.. code-block:: yaml

   # my-facility-dev.yml
   name: "My Facility (Dev)"
   data_bundle: control_assistant
   provider: anthropic
   model: sonnet
   channel_finder_mode: in_context
   requires_osprey_version: ">=2026.5.0"

   config:
     control_system.type: mock
     system.timezone: America/New_York

.. code-block:: bash

   osprey build my-facility my-facility-dev.yml -o /tmp --force

This renders the ``control_assistant`` template with a mock control system and produces
a complete Osprey agent project at ``/tmp/my-facility/``.


Profile YAML Schema
====================

.. list-table::
   :header-rows: 1
   :widths: 20 12 15 53

   * - Field
     - Type
     - Default
     - Description
   * - ``name``
     - string
     - *required*
     - Human-readable profile name.
   * - ``data_bundle``
     - string
     - ``control_assistant``
     - App template (data bundle) to use. Valid: ``control_assistant``,
       ``hello_world``, ``ariel_standalone``. See ``src/osprey/templates/apps/``.
   * - ``provider``
     - string
     - *required*
     - LLM provider for the agent. Built-ins: ``anthropic``, ``cborg``,
       ``als-apg``; any custom provider declared under ``api.providers``
       also works. The build aborts if the resolved profile sets no
       provider (presets already set one).
   * - ``model``
     - string
     - ``None``
     - Default model for the agent: a tier name (``haiku``, ``sonnet``,
       ``opus``) or a full provider model ID (e.g. ``claude-haiku-4-5``).
   * - ``channel_finder_mode``
     - string
     - ``None``
     - Channel finder pipeline (``hierarchical``, ``middle_layer``, ``in_context``).
   * - ``config``
     - mapping
     - ``{}``
     - Dot-notation overrides for ``config.yml``.
   * - ``overlay``
     - mapping
     - ``{}``
     - File/directory overlays (source → destination).
   * - ``mcp_servers``
     - mapping
     - ``{}``
     - MCP server definitions to inject.
   * - ``services``
     - mapping
     - ``{}``
     - Container service definitions for ``osprey deploy`` (see :ref:`profile-services`).
   * - ``lifecycle``
     - mapping
     - ``{}``
     - Commands to run at build phases (``pre_build``, ``post_build``, ``validate``).
   * - ``env``
     - mapping
     - ``{}``
     - Environment variable template (``required``, ``defaults``, ``file``).
   * - ``dependencies``
     - list
     - ``[]``
     - Python packages to append to ``requirements.txt``.
   * - ``requires_osprey_version``
     - string
     - ``None``
     - PEP 440 version specifier (e.g. ``>=2026.5.0``). Build aborts if not satisfied.
   * - ``osprey_install``
     - string
     - ``local``
     - How to install OSPREY in the project venv: ``local`` (source tree), ``pip`` (PyPI), or a PEP 508 spec.
   * - ``python_env``
     - string
     - ``project``
     - Python used by MCP servers: ``project`` (project venv), ``build`` (build-time Python), or an absolute path.


Configuration Overrides
=======================

The ``config:`` section uses **dot notation** to override any key in the generated
``config.yml``. The base set of keys is in
``src/osprey/templates/project/config.yml.j2``; app data bundles add further
sections (e.g. ``archiver``, ``channel_finder``) in their own ``config.yml.j2``
under ``src/osprey/templates/apps/<bundle>/``.

.. warning::

   Always write overrides as **dotted keys**, one per line, exactly as shown
   below — never as nested YAML. A nested block counts as *one* override whose
   value replaces the entire subtree in the rendered config. For example,
   ``config: {claude_code: {model: opus}}`` wipes out everything else under
   ``claude_code`` (servers, permissions, …), silently. The dotted form
   ``claude_code.model: opus`` changes just that one setting.

.. code-block:: yaml

   config:
     # Control system
     control_system.type: epics
     control_system.writes_enabled: true
     control_system.limits_checking.enabled: true
     control_system.connector.epics.timeout: 10.0

     # Archiver
     archiver.type: epics_archiver
     archiver.epics_archiver.url: https://archiver.facility.org

     # System
     # Set your real facility zone: it governs how the agent reads operator
     # times (parsed as facility-local) and renders every timestamp (with an
     # explicit offset) — not just a display label.
     system.timezone: America/Los_Angeles

     # Channel finder
     channel_finder.pipeline_mode: middle_layer

     # Container runtime
     container_runtime: podman

     # Approval policy
     approval.default_policy: always


File Overlays
=============

Overlays copy facility-specific files into the built project, replacing template
defaults. Keys are source paths relative to the profile YAML directory; values are
destination paths relative to the project root.

.. code-block:: yaml

   overlay:
     data/channels.json: data/channel_databases/channels.json
     data/limits.json: data/channel_limits.json
     mcp_servers/custom: _mcp_servers/custom
     prompts/facility.md: .claude/rules/facility.md

Common overlay targets:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Purpose
     - Destination
     - Notes
   * - Channel database
     - ``data/channel_databases/{name}.json``
     - Replaces template example
   * - Channel safety limits
     - ``data/channel_limits.json``
     - Min/max/step per channel
   * - Custom MCP server
     - ``_mcp_servers/{name}/``
     - Directory copy
   * - Facility rule
     - ``.claude/rules/{name}.md``
     - Custom agent rule
   * - Benchmark data
     - ``data/benchmarks/{name}.json``
     - Evaluation datasets

.. admonition:: Path Safety
   :class: important

   Overlay destinations must be **relative paths** without ``..`` components. Absolute
   paths and path traversal are blocked. Source paths that don't exist on disk cause a
   validation error at load time.


MCP Server Injection
====================

Custom MCP servers are recorded in the project's ``config.yml`` (under
``claude_code.servers``) and rendered from there into ``.mcp.json`` (server
configuration) and ``.claude/settings.json`` (tool permissions) — so a later
``osprey claude regen`` re-renders them instead of losing them.

.. code-block:: yaml

   mcp_servers:
     my_server:
       command: python
       args: ["-m", "my_server"]
       env:
         CONFIG: "{project_root}/config.yml"
         API_KEY: "${MY_API_KEY}"
       permissions:
         allow: ["safe_tool"]
         ask: ["write_tool"]

**Placeholder resolution:**

- ``{project_root}`` — resolved at **build time** to the absolute project path
- ``${ENV_VAR}`` — preserved for **runtime** resolution (not expanded during build)

**Permission wiring:** For a server named ``my_server`` with ``allow: ["safe_tool"]``,
the build adds ``mcp__my_server__safe_tool`` to the allow list in
``.claude/settings.json``.

The recommended pattern for facility MCP servers:

1. Write the server as a standalone Python package with ``__main__.py``
2. Place it in ``mcp_servers/{name}/`` in your profiles repo
3. Overlay it to ``_mcp_servers/{name}/`` in the project
4. Set ``PYTHONPATH: "{project_root}/_mcp_servers"`` so ``python -m {name}`` resolves

.. code-block:: yaml

   # Two-step wiring: overlay copies code, mcp_servers entry launches it
   overlay:
     mcp_servers/phoebus: _mcp_servers/phoebus

   mcp_servers:
     phoebus:
       command: python
       args: ["-m", "phoebus"]
       env:
         OSPREY_CONFIG: "{project_root}/config.yml"
         PYTHONPATH: "{project_root}/_mcp_servers"
       permissions:
         allow: ["phoebus_launch"]


.. _profile-tool-permissions:

Tool Permissions
================

By default OSPREY writes a deny list into ``.claude/settings.json`` that blocks a
handful of general-purpose tools — ``Bash``, ``Edit``, ``WebFetch``, ``WebSearch``,
and the Playwright/Context7 plugins — so a stock control-operator agent cannot shell
out or browse the web. These defaults are **overridable per facility** from your
profile's ``config:`` section, using dotted keys (merged on top of the framework
defaults at build time). A top-level ``claude_code:`` block in a profile is *not*
a recognized profile field and would be ignored with a warning — always go
through ``config:``:

.. code-block:: yaml

   config:
     claude_code.permissions.remove_deny: ["Bash", "WebSearch"]  # drop from the default deny list
     claude_code.permissions.allow: ["WebSearch"]                # then allow outright
     claude_code.permissions.ask: ["Bash"]                       # or route to human approval

Supported keys:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Effect
   * - ``remove_deny``
     - Remove entries from the built-in deny defaults (e.g. unblock ``Bash``)
   * - ``deny``
     - Add facility-specific deny entries
   * - ``allow``
     - Add allow entries (no approval prompt)
   * - ``ask``
     - Add entries that route through human approval
   * - ``remove_ask``
     - Remove entries from the ask list

.. admonition:: Deny wins, and it wins at runtime too
   :class: important

   Claude Code resolves permissions as **deny > ask > allow**, and a static ``deny``
   entry cannot be overridden during a session. While a tool sits in the deny list,
   an in-session ``/permissions`` "allow once" will **not** unblock it — you must
   ``remove_deny`` it and rebuild. Use ``ask`` instead of ``deny`` for tools you want
   gated but still reachable on a per-call basis.


.. _profile-services:

Services
========

The ``services`` section defines facility-specific containers that ``osprey deploy``
will manage alongside OSPREY's built-in services (e.g. PostgreSQL).

Each service points to a template directory containing a ``docker-compose.yml.j2``
template. The template directory is copied into the project's ``services/`` tree, and
the service is registered in ``config.yml`` under ``services`` and
``deployed_services``.

.. code-block:: yaml

   services:
     typesense:
       template: services/typesense     # Relative to profile directory
       config:
         port: 8108
         api_key: "${TYPESENSE_API_KEY}"

The ``template`` directory must contain at least ``docker-compose.yml.j2``. Optional
``config`` values are written to ``config.yml`` under ``services.<name>``.


Lifecycle Commands
==================

Lifecycle commands run shell commands at three phases of the build pipeline:

- **pre_build** — runs before template rendering (cwd defaults to profile directory)
- **post_build** — runs after git init (cwd defaults to project directory)
- **validate** — advisory checks that warn but don't abort (cwd defaults to project directory)

.. code-block:: yaml

   lifecycle:
     pre_build:
       - name: "Check dependencies"
         run: "pip check"
     post_build:
       - name: "Build search index"
         run: "python scripts/build_index.py"
         cwd: "data"
       - name: "Run integration tests"
         run: "pytest tests/ --junitxml={project_root}/check_results.xml"
         timeout: 300
         stream: true
     validate:
       - name: "Smoke test"
         run: "python -c 'import osprey; print(osprey.__version__)'"

Each step requires ``name`` and ``run``. Optional fields:

- ``cwd`` — resolved relative to the phase default directory.
- ``timeout`` — seconds before the step is killed (default: 120).
- ``stream`` — if ``true``, stdout is printed in real time instead of captured. Can also
  be enabled for all steps via the ``--stream`` CLI flag.

The ``{project_root}`` placeholder is replaced with the built project's absolute path.

Shell metacharacters (``|``, ``&&``, ``||``, ``$(``, backticks) trigger shell execution;
simple commands use ``shlex.split()`` for safer argument handling.

The project venv's ``bin/`` directory is prepended to ``PATH``, so ``python`` and
``pytest`` inside lifecycle commands resolve to the project's Python (with profile
dependencies installed), not the OSPREY build environment.


Environment Templates
=====================

The ``env`` section generates a ``.env.template`` file in the built project, reminding
users which environment variables to set.

.. code-block:: yaml

   env:
     required:
       - API_KEY
       - DB_HOST
     defaults:
       LOG_LEVEL: info
       PORT: "8080"

This produces a ``.env.template`` with:

.. code-block:: text

   # Required
   API_KEY=
   DB_HOST=

   # Defaults
   LOG_LEVEL=info
   PORT=8080

Required variable names must match ``^[A-Z_][A-Z0-9_]*$``.

To ship a pre-populated ``.env`` file (e.g. for non-secret defaults), use the ``file``
key. The path is relative to the profile directory:

.. code-block:: yaml

   env:
     file: envs/dev.env        # Copied to .env in the built project
     required: [SECRET_KEY]     # .env.template is still generated


Dependencies
============

The ``dependencies`` list appends Python package specifiers to the built project's
``requirements.txt``. This ensures facility-specific packages are tracked alongside
framework dependencies.

.. code-block:: yaml

   dependencies:
     - numpy>=1.24
     - pandas
     - scipy~=1.11

The build installs these into the project venv automatically; only builds run
with ``--skip-deps`` need a manual ``pip install -r requirements.txt``.


Repository Structure
====================

A facility profiles repository should follow this layout:

.. code-block:: text

   my-profiles/
   ├── .gitignore
   ├── facility-dev.yml               # Dev profile (mock control system)
   ├── facility-prod.yml              # Production profile (real hardware)
   ├── data/
   │   ├── channels.json              # Channel database
   │   ├── channel_limits.json        # Safety limits
   │   └── benchmarks/
   │       └── pv_finder_benchmark.json
   ├── prompts/                       # Facility-specific agent rules
   │   ├── facility.md
   │   └── domain-knowledge.md
   └── mcp_servers/                   # Custom MCP server packages
       └── my_server/
           ├── __init__.py
           ├── __main__.py
           ├── server.py
           └── tools/
               ├── __init__.py
               └── my_tool.py

This repository is consumed by ``osprey build`` but kept separate from the OSPREY
framework itself — any facility can create their own equivalent.


CLI Reference
=============

.. code-block:: text

   osprey build PROJECT_NAME [PROFILE] [OPTIONS]

**Arguments:**

- ``PROJECT_NAME`` — name of the project directory to create
- ``PROFILE`` — *optional* path to a YAML build profile. Mutually exclusive
  with ``--preset``; exactly one of the two must be provided.

**Options:**

.. list-table::
   :widths: 30 70

   * - ``--preset NAME``
     - Use a bundled preset profile instead of a positional ``PROFILE`` path.
       Run ``osprey build --list-presets`` to see what ships.
   * - ``-O, --override FILE``
     - Layer a YAML override file on top of the base profile/preset. May be
       repeated; files apply in declaration order. Top-level keys deep-merge;
       string lists union-dedup.
   * - ``--set KEY.PATH=VALUE``
     - Inline scalar/list override. RHS is parsed as YAML, so
       ``--set tier=3`` lands an int and
       ``--set hooks=[memory-guard]`` lands a list. May be repeated.
       ``--set`` wins over ``-O`` files at the same key.
   * - ``--list-presets``
     - List bundled preset names and exit (eager — no ``PROJECT_NAME`` needed).
   * - ``-o, --output-dir DIR``
     - Output directory (default: current directory).
   * - ``-f, --force``
     - Re-render an existing project directory in place. Everything
       framework-owned is rebuilt; ``.env``, ``_agent_data/``, and ``.git``
       are preserved.
   * - ``--tier {1,3}``
     - Channel-database tier: selects which
       ``data/channel_databases/tiers/tier{N}/`` database the rendered config
       points at. Advanced — overrides the default derived from the channel
       finder paradigm (``in_context`` → tier 1, ``hierarchical`` /
       ``middle_layer`` → tier 3). Tier 1 is ``in_context``-only.
   * - ``--emit-profile DIR``
     - Scaffold an editable profile directory at ``DIR`` that extends
       ``--preset``, then exit without rendering a project. Build from it
       with ``osprey build <PROJECT_NAME> DIR/profile.yml``.
   * - ``-s, --stream``
     - Stream lifecycle step output in real time.
   * - ``--skip-lifecycle``
     - Skip ``pre_build``, ``post_build``, and ``validate`` phases.
   * - ``--skip-deps``
     - Skip venv creation and dependency installation (CI mode).
   * - ``--runtime-root PATH``
     - Override ``project_root`` in the rendered config. Useful when the
       build path differs from the runtime path (e.g. container builds).

**Layer ordering:** base preset/profile → ``-O`` override file(s) in
declaration order → ``--set`` pairs.

**Examples:**

.. code-block:: bash

   # See what presets ship
   osprey build --list-presets

   # Build from a bundled preset
   osprey build my-assistant --preset hello-world

   # Build from a profile file
   osprey build als-test ~/als-profiles/als-dev.yml

   # Layer overrides on a preset
   osprey build als-test --preset control-assistant \
       -O als-overrides.yml \
       --set model=claude-sonnet-4-6


Build Pipeline
==============

When ``osprey build`` runs, it executes these steps in order:

1. **Load and validate** the YAML profile (schema check, path existence)
2. **Check version constraint** — abort if ``requires_osprey_version`` is not satisfied
3. **Resolve output path** and handle ``--force`` (re-render in place,
   preserving ``.env``, ``_agent_data/``, and ``.git``)
4. **Run pre_build commands** (cwd: profile directory)
5. **Clear Osprey agent state** for the target directory
6. **Create project venv** — install OSPREY (per ``osprey_install``) and profile dependencies
7. **Build context** from profile fields (provider, model, python_env, channel finder mode)
8. **Render base template** via ``TemplateManager.create_project()``
9. **Apply config overrides** using dot-notation → nested key updates
10. **Copy service templates** (built-in containers for ``osprey deploy``)
11. **Inject profile services** (facility containers from ``services:``)
12. **Copy overlay files** from the profile directory into the project
13. **Inject MCP servers** — recorded in ``config.yml`` under
    ``claude_code.servers``, then rendered into ``.mcp.json`` and
    ``.claude/settings.json``
14. **Copy .env file** (if ``env.file`` is set)
15. **Generate .env.template** from ``env.required`` and ``env.defaults``
16. **Generate manifest** (``.osprey-manifest.json``) for migration tracking
17. **Initialize git** and create an initial commit
18. **Run post_build commands** (cwd: project directory)
19. **Run validate commands** (advisory, cwd: project directory)

.. note::

   The project venv (step 6) is created **before** template rendering so that
   templates can reference the resolved Python path. Lifecycle commands in
   ``post_build`` and ``validate`` automatically use the project venv's Python.

The generated project contains everything the Osprey agent needs to run — no dependency on
the profiles repository at runtime.


What Gets Generated
===================

After building, the project contains:

.. code-block:: text

   built-project/
   ├── .claude/
   │   ├── agents/           # From preset profile (channel-finder, data-visualizer, ...)
   │   ├── rules/            # From preset profile (safety, error-handling, ...)
   │   ├── hooks/            # From preset profile (approval, writes-check, limits, ...)
   │   ├── skills/           # From preset profile (diagnose, session-report, ...)
   │   ├── output-styles/    # From preset profile (control-operator)
   │   └── settings.json     # Permissions, hooks, model config
   ├── .mcp.json             # MCP server configurations
   ├── CLAUDE.md             # Generated system prompt
   ├── config.yml            # Config with overrides applied
   ├── data/                 # Template data + overlays
   ├── _mcp_servers/         # Custom server code (from overlays)
   └── ...

Which agents, rules, hooks, and skills are included starts from the
bundled preset profile (``src/osprey/profiles/presets/<preset>.yml``),
but your user profile can customize it: ``agents:``, ``rules:``,
``hooks:``, and ``skills:`` are first-class profile fields, and
``extends:`` union-merges your additions with the preset's set. To
drop a preset artifact entirely, subtract it with ``exclude:`` (see
:ref:`profile-exclude`), build from a standalone profile, or edit the
preset directly.


Troubleshooting
===============

**"Profile 'name' is required"** — Add a ``name:`` field to your profile YAML.

**"Overlay source not found"** — Check that the source path exists relative to the
profile YAML's directory, not the current working directory.

**"Overlay destination must be relative without '..'"** — Destination paths cannot
be absolute or contain ``..``.

**"MCP server 'X' missing 'command' or 'url'"** — Every MCP server definition
needs a ``command`` (stdio) or a ``url`` (HTTP) field.

**"Directory 'X' already exists"** — Use ``--force`` to rebuild it in place
(your ``.env``, ``_agent_data/``, and ``.git`` are preserved), or pick a
different project name.

**"OSPREY X does not satisfy requires_osprey_version"** — Upgrade OSPREY to a
version matching the profile's specifier, or remove the constraint.

**"Service 'X' template dir missing docker-compose.yml.j2"** — The service template
directory must contain a ``docker-compose.yml.j2`` file.


.. seealso::

   :doc:`../cli-reference/index`
       Complete CLI command reference

   :doc:`add-mcp-server`
       How to build custom MCP servers for OSPREY

   :doc:`deploy-project`
       Container deployment after building
