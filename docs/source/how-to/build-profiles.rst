.. _how-to-build-profiles:

==============
Build Profiles
==============

A **build profile** is the directory your facility owns: one ``profile.yml``, your
data tree, your secrets, and any rules, skills or scripts you write. ``osprey build``
reads that directory and renders a project from it.

The profile is the source of truth. The project is a derived artifact — regenerable,
and safe to delete and rebuild at any time.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Creating a profile directory and building projects from it
   - What lives in a profile: the convention directories, ``data/``, secrets, personas
   - Moving an artifact you want to own out of the project and into the profile
   - Shipping and wiring your own hook scripts
   - Keeping a profile and its project in step

   **Prerequisites:** A working OSPREY installation (``uv sync``).

   **Time:** 15--30 minutes for a basic profile.


Preset → Profile → Project
==========================

.. mermaid::

   flowchart LR
      P["Preset<br/>(bundled with OSPREY)"] -- materialize --> F["Profile directory<br/>(yours)"]
      F -- osprey build --> J["Project<br/>(derived)"]
      J -- osprey deploy --> R["Running containers"]

- **Preset** — a bundled starting point, shipped inside OSPREY
  (``src/osprey/profiles/presets/``). Examples: ``hello-world``,
  ``control-assistant``, ``ariel-standalone``, ``channel-finder-standalone``.
  Run ``osprey profile presets`` to list them.
- **Profile** — your facility's directory. Created once from a preset, then
  edited and kept in version control. Everything the preset configured is
  written out here explicitly: nothing is inherited at build time.
- **Project** — the output of ``osprey build``. Never edit it in place; the next
  build overwrites what you changed.

Because nothing is inherited, a later OSPREY release that improves a preset does
**not** change your profile. To see what moved, materialize a fresh profile into
a scratch directory and diff it:

.. code-block:: bash

   osprey profile new /tmp/fresh --preset control-assistant
   diff -u /tmp/fresh/profile/profile.yml my-facility/profile/profile.yml


Creating a profile
==================

Two commands get you a profile. They differ in what surrounds it.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Command
     - What it does
   * - ``osprey profile new my-facility --preset X``
     - Writes a **facility repository** and stops: a git repository whose
       ``profile/`` directory holds the profile, with an empty ``build/`` for
       rendered projects beside it. Use it when you want to look at and edit
       the profile before building anything.
   * - ``osprey build my-project --preset X``
     - Writes a bare ``my-project-profile/`` directory beside the project on
       the *first* run, then builds from it. Every later run reuses that
       directory as it stands.

Either way the profile itself is the same. After it exists, the everyday
command names it directly:

.. code-block:: bash

   osprey build my-project my-profile/profile.yml

A profile nested in a facility repository is the one case where the build
chooses the output directory for you: it renders into that repository's
``build/<PROJECT_NAME>/``, from whichever directory you run the command.

.. admonition:: Every build reads a profile
   :class: important

   There is no build that renders a project straight out of a bundled preset.
   ``--preset`` on a project that already has a profile directory reuses that
   directory verbatim — the preset is not re-applied. Naming a *different*
   preset for an existing profile is refused: the profile would be built either
   way, and the label would lie about what was built.

   A preset that has changed since your profile was made is reported as a
   warning, never re-applied.

What ``osprey profile new`` writes
----------------------------------

.. code-block:: text

   my-facility/
     profile/
       profile.yml     the full configuration — edit freely
       data/           facility content: channel databases, knowledge, lattice
       .env.example    every variable the agent reads, documented, no values
       .env            your values (only when your shell had keys to seed)
       .gitignore      keeps .env out of version control
       README.md       explains the layout, for whoever opens the directory next
       triggers.yml    the events the agent runs on (dispatch profiles only)
       personas/       one delta per web-terminal persona (persona presets only)
       web-terminal-context/  one seeded directory per operator on the roster
     build/            empty; where `osprey build` renders projects
     ci-extra.yml      the facility's own CI jobs; never regenerated
     .gitignore        keeps build/ and the profile's secrets out of git

The last three files under ``profile/`` appear only when the preset calls for
them — a ``hello-world`` profile has none of them.

``git init`` runs at the repository root, and nothing is committed. There is no
CI pipeline yet: the profile ships its ``deploy:`` block commented out, so
there are no coordinates to render one from. Fill the block in and
``osprey deploy scaffold`` writes the pipeline — see
:doc:`deploy-a-facility`.

Directories for your own artifacts (``rules/``, ``skills/``, and the rest) are
**not** created up front. Create the ones you need; a directory you never create
simply means the profile contributes nothing of that kind.


Convention directories
======================

Put a file in the directory that matches what it is, and the build carries it
into the project. There is nothing to declare in ``profile.yml``: the directory
name *is* the declaration, and where each one lands is fixed.

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Put it here
     - It lands here
     - One entry is
   * - ``rules/``
     - ``.claude/rules/``
     - a ``.md`` file
   * - ``skills/``
     - ``.claude/skills/``
     - a directory with a ``SKILL.md``
   * - ``agents/``
     - ``.claude/agents/``
     - a ``.md`` file
   * - ``commands/``
     - ``.claude/commands/``
     - a ``.md`` file
   * - ``output-styles/``
     - ``.claude/output-styles/``
     - a ``.md`` file
   * - ``hooks/``
     - ``.claude/hooks/``
     - a script, usually ``.py``
   * - ``web-terminal-context/``
     - ``docker/web-terminal-context/``
     - a directory named for one operator
   * - ``mcp_servers/``
     - ``_mcp_servers/``
     - a directory per server
   * - ``services/``
     - ``services/``
     - a directory per compose service
   * - ``project/``
     - the project root
     - any file, mirrored verbatim

Nested paths inside a markdown directory are preserved, so
``commands/orbit/correct.md`` stays namespaced. Skills, MCP servers, services
and per-user context copy as whole directories — the directory *is* the entry,
and a build replaces it wholesale.

A file you ship this way is registered as **yours** in the project: later
re-renders never overwrite it, and cleanup never removes it. Name a file after
something the framework also renders (``rules/safety.md``) and yours wins.

Ownership is *derived* from what the build actually copied, after exclusions
are applied — there is no list to maintain. An artifact a persona delta
excludes is not copied and therefore not owned, so the framework's own version
renders in its place.

.. admonition:: A misspelled directory is silent
   :class: important

   ``rule/`` is not ``rules/``, and nothing reads it. The build warns about
   unrecognized top-level entries in a profile for exactly this reason — read
   that warning rather than wondering why an artifact never arrived.

Paths the profile may not write
-------------------------------

``project/`` is the escape hatch for anything without a home in the table above.
It cannot write paths the build already owns, because each of those has its own
channel:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Path
     - Written by
   * - ``config.yml``
     - the profile's ``config:`` block
   * - ``.claude/settings.json``
     - ``config:`` keys — ``claude_code.permissions``, ``claude_code.hooks``
   * - ``.claude/hooks/hook_config.json``
     - the build, from your ``mcp_servers:`` and ``control_system.write_tools``
   * - ``.mcp.json``
     - the profile's ``mcp_servers:`` block
   * - ``CLAUDE.md``
     - the profile's ``claude_md_template:`` key
   * - ``.env`` / ``.env.example``
     - the profile's own ``.env`` file and ``env:`` keys
   * - ``.osprey-manifest.json``
     - the build itself
   * - ``data/simulation/channel_manifest.json``, ``channel_limits.json``
     - the profile's ``data/`` directory

A profile that targets one of these is rejected at build time, with the owning
channel named. The same refusal applies to a claim (below).

``hook_config.json`` is the one worth understanding: the write-safety hook reads
it to decide what counts as a hardware write. A hand-written copy would be
treated as yours and never regenerate, quietly freezing that decision.


.. _profile-claim:

Taking ownership of a framework artifact
========================================

To customize something OSPREY generates — a rule, an agent, a service template
— move it into the profile:

.. code-block:: bash

   cd my-project
   osprey scaffold claim rules/safety
   osprey scaffold claim agents/channel-finder
   osprey scaffold claim services/postgresql

The artifact is **moved** into the matching convention slot of the profile the
project was built from. Edit it there, then rebuild:

.. code-block:: bash

   osprey build my-project my-profile/profile.yml --force

The next build copies it back and registers it as yours. There is no YAML to
edit — ownership is derived from what the build copied, not declared.

.. code-block:: bash

   osprey scaffold list                     # what is framework-managed, what is yours
   osprey scaffold diff rules/safety        # how far your copy has drifted
   osprey scaffold unclaim rules/safety     # give it back to the framework

``unclaim`` holds only until the next build: while the profile still supplies the
file, the build copies it in and registers it again. To give an artifact up for
good, delete it from the profile.

A claim is refused, with the reason, when:

- the project names no profile to claim into (nothing would keep the edit);
- the artifact is **generated**, not authored — ``hook_config.json``,
  ``settings.json``, ``.mcp.json``, ``CLAUDE.md``. The message names the config
  key that *does* control it;
- the file is a symlink pointing outside the project (a profile must be
  self-contained to be reproducible);
- the profile slot is already occupied. A claim never overwrites profile
  material.

Before reaching for a claim, check whether a config key already covers your need
— most service knobs (ports, images, credentials, retention) are configurable
without owning the template.


Custom hooks
============

A hook is a script the agent runs at a defined moment — before a tool call, at
session start. Ship yours through ``hooks/``:

.. code-block:: text

   my-profile/
     hooks/
       facility_guard.py

That copies the script to ``.claude/hooks/facility_guard.py``. **It does not make
the script run.** Shipping and wiring are two steps; the second is a ``config:``
key naming the event it fires on:

.. code-block:: yaml

   config:
     claude_code.hooks.PreToolUse:
       - hook: facility_guard.py
         matcher: "mcp__controls__.*"   # optional — defaults to every tool call
         timeout: 10                    # optional — seconds, defaults to 60
     claude_code.hooks.SessionStart:
       - facility_banner.py             # shorthand when there is nothing to qualify

Valid events: ``PreToolUse``, ``PostToolUse``, ``UserPromptSubmit``,
``SessionStart``, ``SessionEnd``, ``Stop``, ``SubagentStop``, ``Notification``,
``PreCompact``.

.. admonition:: An undeclared hook never runs
   :class: important

   Without a declaration the script still lands in ``.claude/hooks/`` and
   survives every rebuild — doing nothing. This matters most for a safety check,
   where "present" looks like "enforcing."

Declared wiring is **added** to the framework's, never put in its place. Your
declaration cannot remove, alter, or displace anything the generated settings
already wire — the write-safety gate and the rest render unchanged. Every hook
whose matcher fits runs, so a declared hook is one *more* check on top of the
framework's, never a substitute for one.

A declaration is refused at build time, with the reason, if it names a hook the
resolved profile does not ship, a built-in hook whose wiring the framework
already owns, or anything outside ``hooks/``.

.. _profile-unwire-hook:

Unwiring a hook in a persona
----------------------------

The wiring is a ``config:`` key, so a persona delta overrides it — but you have
to use the right spelling:

.. code-block:: yaml

   config:
     claude_code.hooks.PreToolUse: null   # this event now wires nothing
     claude_code.hooks: {}                # or: unwire every event at once

Either form leaves the script itself in place: unwired, not unshipped.

.. admonition:: An empty list does nothing
   :class: important

   Persona lists merge **additively** with the profile's, so
   ``claude_code.hooks.PreToolUse: []`` adds no entries and leaves the hook
   wired — silently. ``null`` is the spelling that works.

   This matters because a persona that ``exclude:``\ s a shipped hook **must**
   unwire it in the same delta: the build refuses a declaration pointing at a
   hook the persona dropped. That refusal prints the exact ``null`` line to
   paste, so if you reach for ``[]`` on an excluded hook the build hands you the
   correction rather than letting it pass.

Replacing a built-in hook
-------------------------

Shipping and declaring behave differently for the framework's own hooks.

**Shipping** a file named for a built-in *replaces* it: ``hooks/`` is keyed by
filename, because that is what the generated settings run. The built-in
``writes-check`` hook is ``osprey_writes_check.py``, so a profile file of that
name is the one the agent runs wherever the framework already wired that name.

**Declaring** a built-in hook is refused. The framework wires its own hooks from
the profile's ``hooks:`` selection, so a declaration naming one would invoke it
twice. Select or unselect a built-in through ``hooks:``; never through
``claude_code.hooks``.


Personas
========

Some presets give each operator their own web terminal, and each terminal runs
with a persona — a capability posture, such as read-only versus write-capable.
For those presets (``control-assistant``), ``osprey profile new`` writes one
file per persona:

.. code-block:: text

   my-profile/
     profile.yml
     data/
     personas/
       readonly.yml       # a read-only terminal
       readwrite.yml      # a write-capable terminal

Each file holds only that persona's **differences** — for the read-only persona,
chiefly ``control_system.writes_enabled: false``. Sitting in ``personas/`` beside
``profile.yml`` is what makes it a persona: the build merges it over that profile
automatically. There is no ``extends:`` line to maintain, and no second data tree
or set of convention directories — everything else comes from the profile above
and stays in one place.

Edit a delta to change what that persona's terminal can do, and see the merged
result with:

.. code-block:: bash

   osprey profile validate my-profile/personas/readonly.yml

``profile.yml`` points at these files by path — its web-terminal catalog carries
``build_profile: personas/<name>.yml`` for each one — so keep the names in step
if you rename one. That is also what ``osprey deploy up`` reads: it renders any
persona project that does not exist yet from the named delta. A bundled preset
name in that field is rejected, because a persona built from a preset of its own
would not share this profile's data tree, secrets or artifacts.


.. _profile-exclude:

Removing something a profile brings
-----------------------------------

``exclude:`` subtracts entries, and the spelling decides what it removes:

.. code-block:: yaml

   exclude:
     skills:
       - writing-bluesky-plans        # bare: stop selecting the built-in skill
     agents:
       - agents/channel-finder        # qualified: drop the profile's own file

A **bare** name unselects a built-in artifact, so it is no longer installed. A
**qualified** name (``<directory>/<name>``) omits the profile's own file for that
name — which is how a persona that wants the stock version back gets it: drop
your shadowing copy, and the still-selected built-in renders again.

Bare exclusion accepts ``skills``, ``rules``, ``hooks``, ``agents``,
``output_styles``, ``web_panels`` and ``dependencies``; qualified exclusion
accepts any convention directory. Excluding something that is not there is a
silent no-op.

Excluding a **declared hook** takes one more line: the same delta has to unwire
it with ``claude_code.hooks.<Event>: null``, or the build refuses the wiring
that now points at a file the persona dropped. See :ref:`profile-unwire-hook`.

.. admonition:: The mistake worth knowing about
   :class: important

   Use the **bare** spelling on a name your profile also ships a file for, and
   the exclusion does nothing visible: the built-in is unselected, but your file
   still renders, so the project comes out byte-identical and the build
   succeeds. The build warns when it sees this and names the qualified spelling
   that would actually drop the file. Read that warning rather than trusting a
   green build as proof the exclusion took.

.. note::

   ``exclude:`` carves a tier by *removing* capability. When the boundary you
   want is "may not write," prefer flipping the enforcement switch instead —
   the bundled ``control-assistant-readonly`` preset differs from its
   write-capable sibling only on ``control_system.writes_enabled``, leaving
   the tool surface identical (see :doc:`multi-user`).

To keep the scan server **on** while hiding an individual plan, set
``bluesky.excluded_plans`` instead:

.. code-block:: yaml

   bluesky:
     excluded_plans: [orm]

The named plan is then invisible to the agent and non-runnable. The same
block's ``plan_dir`` key does the opposite — it installs a directory of your
facility's own scan plans; see :doc:`bluesky/write-plans`.


.. _profile-secrets:

Secrets
=======

API keys and service credentials belong to the profile directory, not to the
projects built from it. ``osprey build`` derives a project's ``.env`` from the
profile's ``.env``, so a value you set once survives every rebuild — and a
project you delete takes no secret with it.

Two files, and the difference matters:

- ``.env.example`` lists every variable the agent reads, with no values. It is
  safe to commit, and it is the file to read when you want to know what can be
  set. The project gets a copy of this same file, so the two can never document
  different variables.
- ``.env`` holds the values. The generated ``.gitignore`` keeps it out of git.

Seeding, once
-------------

Materializing a profile — ``osprey profile new``, or the first ``--preset``
build, which materializes one — seeds the new profile's ``.env`` from your
shell, and only the keys of providers this profile actually references. Keys
you exported for other providers are named in the summary rather than copied
in, so you can tell "seen and not needed" from "lost". If your shell exported
nothing usable, no ``.env`` is written at all (an empty secrets file reads as a
configured one); start it yourself:

.. code-block:: bash

   cp my-profile/.env.example my-profile/.env

.. admonition:: This is the only moment a shell export reaches a profile
   :class: important

   It happens **once**, at materialization, and what it took is written into
   the profile's ``.env`` under a "Seeded by ``osprey profile new`` from your
   shell" heading — so the file itself records where each value came from.
   Nothing else in the pipeline reads your environment for secrets:
   ``osprey build`` derives the project's ``.env`` from the profile and from
   nothing else, and a later build never re-reads your shell. A key that
   reaches a built project was recorded in the profile first, where you can
   audit it.

   The practical consequence: exporting a key *after* the profile exists does
   not get it in. Put it in the profile's ``.env`` and rebuild.

Service credentials
-------------------

``osprey deploy up`` mints the credentials that only a deploy can produce —
database passwords, service tokens — and writes them back into the profile
``.env`` under a "Minted by deploy" heading. The project's ``.env`` is then
re-derived from the profile, so a rebuild comes up on the *same* secrets instead
of minting a second set the running containers do not trust.

The write-back is **append-only**. A key already in the profile keeps its value —
it is pinned by the docker volume that was initialized with it — and a value that
disagrees is reported by name (never by value) for you to resolve by hand.

If the profile cannot be reached — it has moved, or the project was built before
this mechanism existed — the deploy still works. The secrets stay in the project
``.env``, a warning names the path that failed, and the project records that its
``.env`` is the only copy. A later ``osprey build --force`` repeats that warning
before touching the directory.


Profile YAML reference
======================

.. list-table::
   :header-rows: 1
   :widths: 22 12 14 52

   * - Field
     - Type
     - Default
     - Description
   * - ``name``
     - string
     - *required*
     - Human-readable profile name.
   * - ``app_template``
     - string
     - ``control_assistant``
     - App template (data bundle) to render. Valid: ``control_assistant``,
       ``hello_world``, ``ariel_standalone``.
   * - ``data``
     - string
     - ``None``
     - Facility data tree, relative to the profile directory (``data`` in a
       materialized profile). Replaces the bundled tree wholesale.
   * - ``provider``
     - string
     - *required*
     - LLM provider. Built-ins: ``anthropic``, ``cborg``, ``als-apg``; any
       provider declared under ``api.providers`` also works. The build aborts
       if none is set.
   * - ``model``
     - string
     - ``None``
     - Default model: a tier name (``haiku``, ``sonnet``, ``opus``) or a full
       provider model ID.
   * - ``channel_finder_mode``
     - string
     - ``None``
     - Channel finder pipeline (``hierarchical``, ``middle_layer``,
       ``in_context``).
   * - ``tier``
     - int
     - derived
     - Channel-database tier (1 or 3). Defaults from the channel finder mode;
       tier 1 is ``in_context``-only.
   * - ``connector``
     - string
     - *from preset*
     - Control-system connector (``mock``, ``virtual_accelerator``, ``epics``,
       …). Shorthand for ``config: {control_system.type: ...}``, so it can be
       set from the command line as ``--set connector=epics``. Setting both
       spellings on one command line is an error rather than a silent
       last-one-wins; a custom connector is still addressed by its dotted
       module path under ``config``.
   * - ``config``
     - mapping
     - ``{}``
     - Dot-notation overrides for the generated ``config.yml``.
   * - ``exclude``
     - mapping
     - ``{}``
     - Entries to subtract from what this profile would otherwise bring
       (see :ref:`profile-exclude`).
   * - ``hooks`` / ``rules`` / ``skills`` / ``agents`` / ``output_styles``
     - list
     - ``[]``
     - Built-in artifacts to install. Your own files go in the matching
       convention directory instead.
   * - ``mcp_servers``
     - mapping
     - ``{}``
     - MCP server definitions to inject.
   * - ``services``
     - mapping
     - ``{}``
     - Container services for ``osprey deploy`` (see :ref:`profile-services`).
   * - ``va_archiver``
     - mapping
     - absent
     - Declares a stored archive for a simulated machine: a MongoDB store and a
       recorder the deploy stands up, seeds and records into
       (see :ref:`profile-va-archiver`).
   * - ``lifecycle``
     - mapping
     - ``{}``
     - Commands to run at build phases (``pre_build``, ``post_build``,
       ``validate``).
   * - ``env``
     - mapping
     - ``{}``
     - Variables the deployment needs: ``required``, ``defaults``, ``file``.
   * - ``dependencies``
     - list
     - ``[]``
     - Python packages to install into the project venv.
   * - ``environment``
     - mapping
     - ``{}``
     - Base interpreter the project environment is built from
       (see :ref:`profile-environment`).
   * - ``requires_osprey_version``
     - string
     - ``None``
     - PEP 440 specifier (e.g. ``>=2026.5.0``). The build aborts if unsatisfied.
   * - ``osprey_install``
     - string
     - ``local``
     - How to install OSPREY in the project venv: ``local``, ``pip``, or a
       PEP 508 spec.
   * - ``python_env``
     - string
     - ``project``
     - Python used by MCP servers: ``project``, ``build``, or an absolute path.
   * - ``provenance``
     - mapping
     - *written*
     - Which preset this profile was materialized from, and that preset's hash.
       Written by the materialization; do not edit it.


Configuration overrides
=======================

The ``config:`` section uses **dot notation** to override any key in the
generated ``config.yml``. The base keys are in
``src/osprey/templates/project/config.yml.j2``; app data bundles add further
sections in their own ``config.yml.j2``.

.. warning::

   Always write overrides as **dotted keys**, one per line — never as nested
   YAML. A nested block counts as *one* override whose value replaces the entire
   subtree. ``config: {claude_code: {model: opus}}`` wipes out everything else
   under ``claude_code`` (servers, permissions, …), silently. The dotted form
   ``claude_code.model: opus`` changes just that setting.

.. code-block:: yaml

   config:
     # Control system
     control_system.type: epics
     control_system.writes_enabled: true
     control_system.limits_checking.enabled: true

     # Archiver
     archiver.type: epics_archiver
     archiver.epics_archiver.url: https://archiver.facility.org

     # Set your real facility zone: it governs how the agent reads operator
     # times (parsed as facility-local) and renders every timestamp — not
     # just a display label.
     system.timezone: America/Los_Angeles

     # Channel finder
     channel_finder.pipeline_mode: middle_layer

     # Approval policy
     approval.default_policy: always


MCP server injection
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

Remote servers declare a ``url`` instead of a ``command``, plus an optional
``transport`` — ``http`` (streamable-HTTP, the default) or ``sse`` (legacy
Server-Sent Events):

.. code-block:: yaml

   mcp_servers:
     matlab:
       transport: http
       url: "http://localhost:8008/mcp"
       permissions:
         allow: ["mml_search"]

``command`` and ``url`` are mutually exclusive, and stdio servers must not set
``transport`` (launching via ``command`` *is* the transport).

**Placeholders:** ``{project_root}`` resolves at build time to the absolute
project path; ``${ENV_VAR}`` is preserved for the container or shell to resolve
at runtime.

**Permission wiring:** for a server named ``my_server`` with
``allow: ["safe_tool"]``, the build adds ``mcp__my_server__safe_tool`` to the
allow list.

Shipping the server's code
--------------------------

Put the package in the profile's ``mcp_servers/`` directory — one directory per
server. The build copies it to ``_mcp_servers/`` in the project, so the launch
command finds it:

.. code-block:: text

   my-profile/
     mcp_servers/
       phoebus/
         __init__.py
         __main__.py
         server.py

.. code-block:: yaml

   mcp_servers:
     phoebus:
       command: python
       args: ["-m", "phoebus"]
       env:
         OSPREY_CONFIG: "{project_root}/config.yml"
         PYTHONPATH: "{project_root}/_mcp_servers"
       permissions:
         allow: ["phoebus_launch"]

The directory name and the ``mcp_servers:`` key are independent: the directory
delivers the code, the key launches it.


.. _profile-tool-permissions:

Tool permissions
================

By default OSPREY blocks a handful of general-purpose tools — ``Bash``,
``Edit``, ``WebFetch``, ``WebSearch``, and the Playwright/Context7 plugins — so a
stock control-operator agent cannot shell out or browse the web. These defaults
are overridable per facility from ``config:``, using dotted keys:

.. code-block:: yaml

   config:
     claude_code.permissions.remove_deny: ["Bash", "WebSearch"]  # drop from the deny list
     claude_code.permissions.allow: ["WebSearch"]                # then allow outright
     claude_code.permissions.ask: ["Bash"]                       # or route to human approval

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Effect
   * - ``remove_deny``
     - Remove entries from the built-in deny defaults
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

   Permissions resolve as **deny > ask > allow**, and a static ``deny`` entry
   cannot be overridden during a session — an in-session "allow once" will not
   unblock it. Use ``ask`` for tools you want gated but still reachable.


.. _profile-services:

Services
========

The ``services`` section defines facility containers that ``osprey deploy``
manages alongside OSPREY's built-in ones.

.. code-block:: yaml

   services:
     typesense:
       template: services/typesense     # relative to the profile directory
       config:
         port: 8108
         api_key: "${TYPESENSE_API_KEY}"

The ``template`` directory must contain at least ``docker-compose.yml.j2``. It is
copied into the project's ``services/`` tree, and the service is registered in
``config.yml``. Optional ``config`` values land under ``services.<name>``.

A service directory placed in the profile's ``services/`` convention directory is
carried across the same way and marked as yours — that is what
``osprey scaffold claim services/<name>`` produces.

.. _profile-va-archiver:

The ``va_archiver`` block
=========================

A deployment that serves simulated channels still needs somewhere to keep what
those channels did. Declaring ``va_archiver:`` is what gives it one: the build
adds a MongoDB store and a recorder to the service stack, ``osprey deploy up``
seeds the store with history and then records the running machine into it, and
the ``mongodb_archiver`` connector reads it back.

.. code-block:: yaml

   va_archiver:
     host: localhost
     retention_days: 30
     hot_span_hours: 48
     hot_cadence_sec: 10
     tail_cadence_sec: 60
     freshness_channel: SR:DIAG:DCCT:01:CURRENT:RB

Every key is optional and the defaults describe a working archive; the block's
presence is the decision, not its contents.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Key
     - Default
     - Meaning
   * - ``retention_days``
     - ``30``
     - How far back the archive reaches — both what a fresh deployment holds
       and what a running one keeps.
   * - ``hot_span_hours``
     - ``48``
     - How much of the recent end is kept at the dense cadence. May not exceed
       ``retention_days``.
   * - ``hot_cadence_sec``
     - ``10``
     - Seconds between samples inside the hot span.
   * - ``tail_cadence_sec``
     - ``60``
     - Seconds between samples outside it. Must be a whole multiple of
       ``hot_cadence_sec`` — the sparse tier is a subset of the dense grid, so a
       cadence that does not divide would put the two on timestamps that never
       coincide.
   * - ``recorder_cadence_sec``
     - ``10``
     - How often the recorder samples the live machine.
   * - ``recorder_tail_cadence_sec``
     - ``60``
     - How often one of those samples is additionally kept for the full
       retention span, so recorded history survives as the dense copy ages out.
       Same whole-multiple rule.
   * - ``recorder_poll_sec``
     - ``30``
     - How often the recorder re-reads the deployment's config to decide whether
       to record at all. It records only for a ``virtual_accelerator`` control
       system, and this is what lets that flip take effect without a restart.
   * - ``freshness_channel``
     - unset
     - Canary channel for a derived ``archiver_freshness`` health check. Unset
       derives no check (see :doc:`configure-health-checks`).
   * - ``host``
     - ``localhost``
     - Where the store is. **Required** when ``deploy_services`` is false: an
       attached project deploys no store of its own, so it has to name the host
       whose archive it reads.
   * - ``port_host``
     - ``27017``
     - Host port the store publishes on — or, for an attached project, the port
       the other host published.
   * - ``database`` / ``collection``
     - ``osprey_archiver`` / ``pv_history``
     - Where the samples live inside the store.
   * - ``compression``
     - ``zstd``
     - Block compressor for the collection: ``zstd``, ``snappy``, ``zlib`` or
       ``none``.
   * - ``username`` / ``auth_database``
     - ``osprey`` / ``admin``
     - The database user the deployment creates and the agent connects as, and
       the database it authenticates against.
   * - ``password_env``
     - ``MONGO_ROOT_PASSWORD``
     - **Name** of the variable holding that password. The value is minted into
       the deployment's ``.env``; it is never a profile field.
   * - ``timeout_sec``
     - ``5``
     - How long the connector waits to reach the store.

One fact, one home
------------------

The block is where the archive is described, and the build writes the rest from
it. Do **not** also spell these in ``config:`` — a profile that does is refused,
by name, rather than silently having one copy win:

- the connector's eight connection keys —
  ``archiver.mongodb_archiver.host``, ``.port``, ``.name``, ``.collection``,
  ``.auth``, ``.username``, ``.password_env``, ``.timeout`` — all derived from
  the keys above;
- the shape knobs, written to ``va_archiver.*`` in the rendered ``config.yml``
  for the seeder and the recorder to read;
- ``health.categories.archiver``, when ``freshness_channel`` is set.

Two homes for one fact are free to disagree, and the disagreement is the
dangerous case: a stale ``collection`` or ``host`` in ``config:`` points the
agent at an archive nothing is writing, which reads as empty rather than as
broken.

What the block does *not* do is select the archiver. Declaring where an archive
lives and choosing it as the deployment's archiver are separate decisions, so
the block never flips ``archiver.type`` out from under you — set
``config: {archiver.type: mongodb_archiver}`` yourself, or the project deploys a
store and then reads something else beside it.

.. warning::

   ``osprey build`` **refuses** a profile that pairs a ``virtual_accelerator``
   control system with the mock archiver, or with no ``archiver.type`` at all
   (which resolves to the mock): a simulated machine whose history is
   synthesized at read time reports a past that never happened, and nothing can
   catch it. The error names the fix — declare this block and select
   ``mongodb_archiver``, point the archiver at a store you run yourself, or set
   the control system to ``mock`` for an honestly storeless project. See
   :doc:`use-virtual-accelerator`.


Lifecycle commands
==================

Lifecycle commands run shell commands at three phases of the build:

- **pre_build** — before rendering (cwd: profile directory)
- **post_build** — after git init (cwd: project directory)
- **validate** — advisory checks that warn but don't abort (cwd: project directory)

.. code-block:: yaml

   lifecycle:
     pre_build:
       - name: "Check dependencies"
         run: "pip check"
     post_build:
       - name: "Build search index"
         run: "python scripts/build_index.py"
         cwd: "data"
         timeout: 300
         stream: true

Each step requires ``name`` and ``run``. Optional: ``cwd`` (relative to the phase
default), ``timeout`` (seconds, default 120), and ``stream`` (print output live;
also available for all steps via ``--stream``).

``{project_root}`` is replaced with the built project's absolute path. The
project venv's ``bin/`` is prepended to ``PATH``, so ``python`` and ``pytest``
resolve to the project's own Python.


Environment variables
=====================

The ``env`` section declares what the deployment needs. It documents variables;
it does not carry values — values live in the profile's ``.env``
(see :ref:`profile-secrets`).

.. code-block:: yaml

   env:
     required:
       - API_KEY
       - DB_HOST
     defaults:
       LOG_LEVEL: info

Both lists are rendered into the profile's ``.env.example``, so an operator
opening that file sees them alongside every other variable. Required names must
match ``^[A-Z_][A-Z0-9_]*$``.

To ship a pre-populated ``.env`` (non-secret defaults, say), use the ``file``
key — a path relative to the profile directory:

.. code-block:: yaml

   env:
     file: envs/dev.env        # copied to .env in the built project


Dependencies
============

``dependencies`` adds Python package specifiers to the built project. They are
installed into the project venv and recorded in its generated ``pyproject.toml``:

.. code-block:: yaml

   dependencies:
     - numpy>=1.24
     - pandas
     - scipy~=1.11

.. code-block:: bash

   cd my-project
   uv run osprey web     # uses my-project/.venv
   uv sync               # rebuilds it from pyproject.toml

Builds run with ``--skip-deps`` create no environment and no ``pyproject.toml``;
install dependencies yourself in that mode.

.. _profile-environment:

The execution environment
-------------------------

``dependencies`` says what *else* to install. The ``environment:`` block says
what the project environment is built *on top of* — which interpreter it starts
from, and, when that interpreter belongs to a virtual environment your facility
already maintains, which of its packages to carry over.

.. code-block:: yaml

   environment:
     python: /opt/facility/analysis-env/bin/python   # base interpreter
     packages:                                       # installed on top
       - lmfit>=1.3
     inherit_exclude:                                # left out of the freeze
       - facility-inhouse-tools

All three keys are optional; the block as a whole can be omitted.

``python``
   The base interpreter, as an absolute path. It may be a plain interpreter
   (``/usr/bin/python3.12``) or the interpreter inside a virtual environment —
   the syntax is the same. The build aborts if the path does not exist or is not
   executable.

``packages``
   Extra requirements installed into the project environment. Resolved in the
   same install as ``dependencies``, so the two cannot disagree; where both name
   the same distribution, a pinned version wins over a bare name, and between two
   pins ``packages`` wins.

``inherit_exclude``
   Distribution names to leave out of the freeze described below. Only meaningful
   with a virtual environment base; declaring it otherwise is rejected at
   validation time rather than silently ignored.

**Carrying a virtual environment's packages over.** Basing a project on a virtual
environment's *interpreter* does not inherit its *packages*. What carries them
over is a **freeze**: when ``environment.python`` names a virtual environment's
interpreter, the build records that environment's installed distributions as
exact ``name==version`` requirements in the project's ``pyproject.toml``. The
project venv — and any container image built from it — installs that same set.

A pin in ``dependencies`` or ``packages`` overrides the version the base
happened to carry.

The freeze runs **only when a base interpreter is declared**. Without
``environment.python`` the base is whatever interpreter OSPREY itself was
installed into — an accident, not a curated environment — and its packages are
deliberately not carried over.

**The build stops if a package cannot be reproduced.** Two cases are refused: a
distribution with no package-index coordinate (installed from a local path, a
VCS checkout, or a bare archive URL), and a version outside OSPREY's own
requirement for that package. Every offending package is named in a single
message, along with the ``inherit_exclude`` block that clears all of them.


Regenerating a channel database
===============================

``osprey channel-finder build-database`` writes the generated database **into the
profile**, not into the project — beside the CSV inputs it came from, where it
survives a rebuild. The sequence is meant to run to completion:

.. code-block:: bash

   osprey channel-finder build-database
   # the project now reports its build as stale
   osprey build my-project my-profile/profile.yml --force
   # the advisory clears

The staleness advisory in between is the reminder that the new database has not
been deployed yet — not a problem to fix. Use ``--output`` to write somewhere
else; a project that resolves no profile falls back to its own ``data/`` tree
and warns that the next build will overwrite it.


Building
========

.. code-block:: text

   osprey build PROJECT_NAME [PROFILE] [OPTIONS]

**Arguments**

- ``PROJECT_NAME`` — name of the project directory to create
- ``PROFILE`` — path to a profile YAML. Mutually exclusive with ``--preset``;
  exactly one of the two is required.

**Options**

.. list-table::
   :widths: 30 70

   * - ``--preset NAME``
     - Materialize ``<PROJECT_NAME>-profile/`` from a bundled preset and build
       from it. Reused as-is if it already exists.
   * - ``-O, --override FILE``
     - Layer a YAML file on top of the profile (repeatable, applied in order).
   * - ``--set KEY.PATH=VALUE``
     - Inline override. The right-hand side is parsed as YAML, so
       ``--set tier=3`` lands an int and ``--set hooks=[memory-guard]`` a list.
   * - ``--tier {1,3}``
     - Channel-database tier. Overrides the default derived from the channel
       finder mode (``in_context`` → tier 1, otherwise tier 3).
   * - ``--list-presets``
     - List bundled preset names and exit.
   * - ``-o, --output-dir DIR``
     - Output directory (default: current directory).
   * - ``-f, --force``
     - Re-render an existing project directory in place.
   * - ``-s, --stream``
     - Stream lifecycle step output in real time.
   * - ``--skip-lifecycle``
     - Skip ``pre_build``, ``post_build``, and ``validate``.
   * - ``--skip-deps``
     - Skip venv creation and dependency installation (CI mode).
   * - ``--runtime-root PATH``
     - Override ``project_root`` in the rendered config, for container builds.

.. admonition:: ``--set``, ``-O`` and ``--tier`` edit the profile
   :class: important

   They are not build-time layers that vanish afterwards. On a build from an
   existing profile they are **written into that profile** before it is read,
   replacing the value at each dotted key — because the profile has to keep
   describing the project. On the first ``--preset`` build they are baked into
   the profile being materialized.

   Comments and formatting are preserved, and a build that then fails puts the
   file back exactly as it was, so a typo never leaves a stuck profile behind.

``--force`` re-renders everything framework-owned and preserves what you own:
``.env``, ``_agent_data/``, and the project's ``.git``. It never touches the
profile — only ``osprey profile new --force`` replaces one of those.

**Examples**

.. code-block:: bash

   # See what presets ship
   osprey build --list-presets

   # First build: materializes my-assistant-profile/ and builds from it
   osprey build my-assistant --preset control-assistant

   # Later builds reuse that profile directory
   osprey build my-assistant --preset control-assistant --force

   # Build from a profile you keep elsewhere
   osprey build als-test ~/als-profiles/als-dev/profile.yml

   # Write a setting into the profile as part of the build
   osprey build als-test ~/als-profiles/als-dev/profile.yml \
       --set model=claude-sonnet-4-6

Checking a profile without building
-----------------------------------

.. code-block:: bash

   osprey profile validate my-profile/
   osprey profile validate my-profile/personas/readonly.yml

Resolves the profile and runs the full consistency check — convention
directories, the data tree, service templates, lifecycle steps, environment
variables — reporting every problem found, not just the first.


What the build does
===================

1. Settle the profile (materialize from a preset on first use, or read the one
   you named), writing any ``--set`` / ``-O`` / ``--tier`` into it.
2. Resolve and validate the profile, including any persona delta merged over it.
3. Check ``requires_osprey_version``; abort if unsatisfied.
4. Handle ``--force`` — clear the rendered files, keeping ``.env``,
   ``_agent_data/`` and ``.git``.
5. Run ``pre_build`` commands.
6. Create the project venv and install OSPREY plus the profile's dependencies.
7. Render the base template and the profile's ``data/`` tree; derive the
   project's ``.env`` from the profile's.
8. Apply the ``config:`` overrides.
9. Copy service templates and inject the profile's own services.
10. Apply the convention directories, and register what was copied as yours.
11. Persist ``mcp_servers:`` into ``config.yml``.
12. Stamp the manifest (``.osprey-manifest.json``), including the profile path
    the deploy later writes secrets back to.
13. Re-render the agent artifacts against the complete config, and validate that
    every tool an agent declares is backed by a permission.
14. Initialize git, then run ``post_build`` and ``validate`` commands.

The venv is created before rendering so templates can reference the resolved
Python path. The generated project runs standalone — nothing reaches back to the
profile at runtime.


What gets generated
===================

.. code-block:: text

   my-project/
   ├── .claude/
   │   ├── agents/           # built-ins, plus anything from the profile's agents/
   │   ├── rules/            # built-ins, plus the profile's rules/
   │   ├── hooks/            # hook scripts, plus the generated hook_config.json
   │   ├── skills/
   │   ├── output-styles/
   │   └── settings.json     # permissions, hook wiring, model config
   ├── .mcp.json             # MCP server configurations
   ├── CLAUDE.md             # generated system prompt
   ├── config.yml            # config with the profile's overrides applied
   ├── data/                 # the profile's data tree, materialized
   ├── _mcp_servers/         # facility server code from the profile
   ├── .env                  # derived from the profile's .env
   └── .env.example          # a copy of the profile's

Which built-in agents, rules, hooks and skills are installed comes from the
``agents:``, ``rules:``, ``hooks:`` and ``skills:`` lists in ``profile.yml``.
Your own files come from the convention directories and are marked as yours.


Troubleshooting
===============

**"Either a profile path or --preset is required"** — every build reads a
profile. Name one, or ``--preset`` to have one materialized.

**"was materialized from preset X, but this build asks for Y"** — the profile
directory beside this project came from a different preset. Build ``Y`` under a
different project name so it gets a profile of its own.

**"Profile convention directories are invalid"** — a convention directory has
the wrong shape: a ``.md`` directory holding something else, a skill that is a
file rather than a directory, or a symlink pointing outside the profile. Every
problem is listed at once.

**"project/ mirror writes N build-owned path(s)"** — the mirror targets a path
another channel owns. The message names the channel, and the exact move where
one exists.

**"Profile has N unrecognized top-level entry/entries"** — a warning, not an
error: a directory in the profile that nothing copies. Usually a typo of a
convention directory name.

**"Unknown profile key(s): 'overlay'"** — profiles no longer have an overlay
section. Move the files into the convention directory that matches what they
are (see the table above), or into ``project/`` for anything without one.

**"Directory 'X' already exists"** — use ``--force`` to rebuild the project in
place, or pick a different project name. ``--force`` never replaces the profile.

**"OSPREY X does not satisfy requires_osprey_version"** — upgrade OSPREY, or
relax the constraint in the profile.


.. seealso::

   :doc:`../cli-reference/index`
       Complete CLI command reference

   :doc:`add-mcp-server`
       How to build custom MCP servers for OSPREY

   :doc:`deploy-project`
       Container deployment after building
