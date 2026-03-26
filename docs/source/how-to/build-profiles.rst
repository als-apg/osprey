.. _how-to-build-profiles:

==============
Build Profiles
==============

Assemble facility-specific assistants from OSPREY templates using declarative YAML profiles.
Build profiles separate **what makes your facility unique** (channel data, safety limits,
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

The ``osprey build`` command takes a YAML profile and produces a standalone Claude Code
project. The profile declares:

- **Base template** to start from (``control_assistant`` or ``lattice_design``)
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


Quick Start
===========

Create a minimal profile and build:

.. code-block:: yaml

   # my-facility-dev.yml
   name: "My Facility (Dev)"
   base_template: control_assistant
   provider: anthropic
   model: sonnet
   requires_osprey_version: ">=0.12.0"

   config:
     control_system.type: mock
     system.timezone: America/New_York

.. code-block:: bash

   osprey build my-facility my-facility-dev.yml -o /tmp --force

This renders the ``control_assistant`` template with a mock control system and produces
a complete Claude Code project at ``/tmp/my-facility/``.


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
   * - ``base_template``
     - string
     - ``control_assistant``
     - App template to use. See ``src/osprey/templates/apps/``.
   * - ``provider``
     - string
     - ``None``
     - LLM provider (``anthropic``, ``cborg``, ``openai``, ``google``, etc.).
   * - ``model``
     - string
     - ``None``
     - Model tier (``haiku``, ``sonnet``, ``opus``).
   * - ``channel_finder_mode``
     - string
     - ``None``
     - Channel finder pipeline (``hierarchical``, ``middle_layer``, ``in_context``, ``all``).
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
     - PEP 440 version specifier (e.g. ``>=0.12.0``). Build aborts if not satisfied.
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
``config.yml``. Available keys are defined in the config template
(``src/osprey/templates/project/config.yml.j2``).

.. code-block:: yaml

   config:
     # Control system
     control_system.type: epics
     control_system.writes_enabled: true
     control_system.limits_checking: true
     control_system.connector.epics.timeout: 10.0

     # Archiver
     archiver.type: epics_archiver
     archiver.epics_archiver.url: https://archiver.facility.org

     # System
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
     - Custom Claude rule
   * - Benchmark data
     - ``data/benchmarks/{name}.json``
     - Evaluation datasets
   * - Example scripts
     - ``_agent_data/example_scripts/{cat}/``
     - Claude learning examples

.. admonition:: Path Safety
   :class: important

   Overlay destinations must be **relative paths** without ``..`` components. Absolute
   paths and path traversal are blocked. Source paths that don't exist on disk cause a
   validation error at load time.


MCP Server Injection
====================

Custom MCP servers are injected into both ``.mcp.json`` (server configuration) and
``.claude/settings.json`` (tool permissions).

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

After building, install with ``pip install -r requirements.txt``.


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
   ├── prompts/                       # Facility-specific Claude rules
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

   osprey build PROJECT_NAME PROFILE [OPTIONS]

**Arguments:**

- ``PROJECT_NAME`` — name of the project directory to create
- ``PROFILE`` — path to a YAML build profile

**Options:**

.. list-table::
   :widths: 25 75

   * - ``--output-dir, -o``
     - Output directory (default: current directory)
   * - ``--force, -f``
     - Overwrite if project directory already exists
   * - ``--stream, -s``
     - Stream lifecycle step output in real time

**Examples:**

.. code-block:: bash

   # Basic build
   osprey build als-test ~/als-profiles/als-dev.yml -o /tmp --force

   # Build with streaming output for lifecycle steps
   osprey build als-test ~/als-profiles/als-dev.yml --stream

   # Build to current directory
   osprey build my-assistant profile.yml


Build Pipeline
==============

When ``osprey build`` runs, it executes these steps in order:

1. **Load and validate** the YAML profile (schema check, path existence)
2. **Check version constraint** — abort if ``requires_osprey_version`` is not satisfied
3. **Resolve output path** and handle ``--force`` (remove existing directory)
4. **Run pre_build commands** (cwd: profile directory)
5. **Clear Claude Code state** for the target directory
6. **Create project venv** — install OSPREY (per ``osprey_install``) and profile dependencies
7. **Build context** from profile fields (provider, model, python_env, channel finder mode)
8. **Render base template** via ``TemplateManager.create_project()``
9. **Apply config overrides** using dot-notation → nested key updates
10. **Copy service templates** (built-in containers for ``osprey deploy``)
11. **Inject profile services** (facility containers from ``services:``)
12. **Copy overlay files** from the profile directory into the project
13. **Inject MCP servers** into ``.mcp.json`` and ``.claude/settings.json``
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

The generated project contains everything Claude Code needs to run — no dependency on
the profiles repository at runtime.


What Gets Generated
===================

After building, the project contains:

.. code-block:: text

   built-project/
   ├── .claude/
   │   ├── agents/           # From manifest (channel-finder, data-visualizer, ...)
   │   ├── rules/            # From manifest (safety, error-handling, ...)
   │   ├── hooks/            # From manifest (approval, writes-check, limits, ...)
   │   ├── skills/           # From manifest (diagnose, session-report, ...)
   │   ├── output-styles/    # From manifest (control-operator)
   │   └── settings.json     # Permissions, hooks, model config
   ├── .mcp.json             # MCP server configurations
   ├── CLAUDE.md             # Generated system prompt
   ├── config.yml            # Config with overrides applied
   ├── data/                 # Template data + overlays
   ├── _mcp_servers/         # Custom server code (from overlays)
   └── ...

Which agents, rules, hooks, and skills are included is controlled by the template's
``manifest.yml`` — not by the profile. The profile can override **data and config** but
not the set of Claude Code artifacts. To add new agents or rules, modify the OSPREY
template (see :doc:`add-mcp-server`).


Troubleshooting
===============

**"Profile 'name' is required"** — Add a ``name:`` field to your profile YAML.

**"Overlay source not found"** — Check that the source path exists relative to the
profile YAML's directory, not the current working directory.

**"Overlay destination must be relative without '..'"** — Destination paths cannot
be absolute or contain ``..``.

**"MCP server 'X' missing 'command'"** — Every MCP server definition needs a
``command`` field.

**"MCP server 'X' already exists in .mcp.json"** — The server name conflicts with
a built-in. Choose a different name.

**"Directory 'X' already exists"** — Use ``--force`` to overwrite, or pick a
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
