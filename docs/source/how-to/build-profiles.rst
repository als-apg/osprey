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
   - Using config overrides and named variants
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
- **Variants** for environment-specific configuration (dev, prod)

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
   * - ``variants``
     - mapping
     - ``{}``
     - Named variant groups for ``--variant``.


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


Named Variants
==============

Variants define environment-specific config overrides applied with ``--variant``.
Variant config is deep-merged onto the base profile (variant wins on conflicts).

.. code-block:: yaml

   name: "My Facility"
   base_template: control_assistant

   config:
     system.timezone: America/Chicago
     channel_finder.pipeline_mode: middle_layer

   variants:
     dev:
       config:
         control_system.type: mock
         control_system.writes_enabled: false
     prod:
       config:
         control_system.type: epics
         control_system.writes_enabled: true
         container_runtime: podman

.. code-block:: bash

   # Build with dev variant
   osprey build my-facility profile.yml --variant dev -o /tmp --force

   # Build with prod variant
   osprey build my-facility profile.yml --variant prod -o /tmp --force

.. tip::

   For simple setups, separate profile files (``facility-dev.yml``,
   ``facility-prod.yml``) are clearer than variants. Variants shine when
   you have many shared settings with small per-environment differences.


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

   * - ``--variant, -v``
     - Apply a named variant from the profile
   * - ``--output-dir, -o``
     - Output directory (default: current directory)
   * - ``--force, -f``
     - Overwrite if project directory already exists

**Examples:**

.. code-block:: bash

   # Basic build
   osprey build als-test ~/als-profiles/als-dev.yml -o /tmp --force

   # Build with variant
   osprey build als-prod ~/als-profiles/base.yml --variant prod --force

   # Build to current directory
   osprey build my-assistant profile.yml


Build Pipeline
==============

When ``osprey build`` runs, it executes these steps in order:

1. **Load and validate** the YAML profile (schema check, path existence)
2. **Resolve output path** and handle ``--force`` (remove existing directory)
3. **Clear Claude Code state** for the target directory
4. **Build context** from profile fields (provider, model, channel finder mode)
5. **Render base template** via ``TemplateManager.create_project()``
6. **Apply config overrides** using dot-notation → nested key updates
7. **Copy overlay files** from the profile directory into the project
8. **Inject MCP servers** into ``.mcp.json`` and ``.claude/settings.json``
9. **Generate manifest** (``.osprey-manifest.json``) for migration tracking
10. **Initialize git** and create an initial commit

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


.. seealso::

   :doc:`../cli-reference/index`
       Complete CLI command reference

   :doc:`add-mcp-server`
       How to build custom MCP servers for OSPREY

   :doc:`deploy-project`
       Container deployment after building
