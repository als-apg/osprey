.. _reference-profile:

===========================
Build Profile — profile.yml
===========================

Every key a build profile's ``profile.yml`` accepts, what the build does with
it, and what the deployment does with the result. For what a profile *is* and
how to write one — the concepts, the worked walkthrough, ``extends``
composition, and troubleshooting — see :doc:`/how-to/build-profiles`.

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
       ``hello_world``, ``ariel_standalone``, ``channel_finder_standalone``.
   * - ``data``
     - string
     - ``None``
     - Facility data tree, relative to the profile directory (``data`` in a
       materialized profile). Replaces the bundled tree wholesale. May resolve
       above the profile directory; only existence and shape are checked
       (see :ref:`profile-self-contained`).
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
       ``in_context``, ``graph``). ``graph`` searches the deployment's graph
       store instead of a channel database, so it needs a ``services.graphdb``
       block (see :ref:`profile-graph-mode`).
   * - ``tier``
     - int
     - derived
     - Channel-database tier (1 or 3). Defaults from the channel finder mode;
       tier 1 is ``in_context``-only. ``graph`` has no tiered artifacts at all —
       leave ``tier`` unset there.
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
     - Container services the deployment runs (see :ref:`profile-services`).
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
     - Variables the deployment needs: ``required``, ``defaults``, ``file``,
       ``pinned``. ``env.pinned`` lists variables ``.env.shared`` decides and a
       host may not override; a deploy refuses rather than start on a
       contradicting value (see :ref:`deployment-pinned-env`). Same name pattern
       as ``required``, and a machine-minted name cannot be pinned.
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
``osprey build`` re-renders them instead of losing them.

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

   my-facility/
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
     claude_code.permissions.remove_deny: ["WebSearch", "WebFetch"]  # drop from the deny list
     claude_code.permissions.allow: ["WebSearch"]                    # then allow outright
     claude_code.permissions.ask: ["WebFetch"]                       # or route to human approval

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Effect
   * - ``remove_deny``
     - Remove entries from the deny list. Matches an entry **exactly** — the
       string has to be the one being denied, character for character.
       Removing a write-capable tool that nothing else gates fails the build —
       see below
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
   unblock it. Use ``ask`` for tools you want gated but still reachable. For the
   same reason, listing a tool under ``ask`` or ``allow`` does nothing until you
   also remove it from the deny list.

.. admonition:: You cannot un-gate a tool that can write
   :class: warning

   ``Bash``, ``Edit``, ``Write``, ``MultiEdit`` and ``NotebookEdit`` can write
   files or shell out, so ``osprey build`` refuses a profile in which one of
   them is neither in ``permissions.deny`` nor matched by a ``PreToolUse`` hook
   matcher. A ``remove_deny: ["Bash"]`` with nothing put in its place is
   therefore a build failure, not a silent widening. (The shipped presets gate
   the three file-writing tools with the ``memory-guard`` hook rather than
   denying them, so ordinary memory and notebook writes still work.)

   The build checks that a covering rule *exists*, not that the hook behind it
   refuses anything: a ``PreToolUse`` hook that exits 0 without a
   ``permissionDecision`` allows the call. A tool whose only cover is a matcher
   your own profile declares therefore builds with a warning — check that hook
   really denies. Matchers are read the way the agent runtime reads them: the
   bare tool name, a ``|`` alternation, any regex that matches the name
   (unanchored), or ``*``/``.*``/empty for every tool.

.. _profile-deny-assembly:

How the deny list is assembled
------------------------------

The ``permissions.deny`` array in the rendered ``.claude/settings.json`` is
built from three sources, in this order:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Source
     - How it is treated
   * - The framework's deny defaults
     - Minus anything the profile lists under ``remove_deny``
   * - The profile's own ``deny``
     - Minus anything the profile lists under ``remove_deny`` — a profile may
       subtract what a profile added
   * - The write kill switch
     - Appended last and **never** filtered

The third source is the one to know about. Setting
``control_system.writes_enabled: false`` does not merely stop the write path
at run time; it also puts the write tools into the rendered deny list. Those
entries are generated, not authored, and no ``remove_deny`` reaches them. A
profile that sets ``writes_enabled: false`` and also writes
``remove_deny: ["mcp__controls__channel_write"]`` still renders that deny —
which is the point. A read-only posture that a later edit could quietly lift
would not be a posture at all.

.. admonition:: Permission lists grow across ``extends``; they never shrink
   :class: important

   The permission lists under ``config:`` — ``deny``, ``ask`` and their
   ``remove_*`` companions — **union** with the ones they inherit. A child
   profile or a persona delta can add to an inherited permission list, but it
   has no way to take an entry out of it: ``exclude:`` subtracts from the
   convention lists (``skills``, ``agents``, ``web_panels`` and their
   siblings — see :ref:`profile-exclude`) and does not reach inside ``config:``
   at all.

   That is why a preset that wants to withhold a privilege from every tier and
   grant it back to one writes the floor as ``deny``, and lets the privileged
   tier lift it with ``remove_deny``. Writing the floor the other way round —
   putting the tool under ``ask`` at the base and expecting a child to
   ``remove_ask`` it — does the opposite of what it looks like: the base's
   ``remove_ask`` would union into *every* child, including the one meant to
   keep the approval prompt, and strip that prompt away. ``deny`` is
   subtractable per tier, which is the direction a floor has to work in.

   The bundled ``control-assistant`` preset is built exactly this way: its
   base denies the agent's own deployment-editing tool, and the ``admin``
   persona is the single delta that removes that deny. Writing the floor is
   not only a convention. A web terminal served without a login by a persona
   that can edit the deployment is refused by ``osprey validate``, ``osprey
   build`` and ``osprey up`` alike, wherever the deployment has a login wall
   at all — with ``auth.method: none`` there is no wall to be exempt from, and
   the same exposure is reported as an advisory instead. On a profile that
   wrote no floor at all, where every persona holds everything, the only
   remedy that refusal can offer is to write one. See
   :doc:`/how-to/web-terminal/multi-user/tiers`.

.. _profile-services:

Services
========

The ``services`` section defines facility containers the deployment runs
alongside OSPREY's built-in ones.

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

One ``config`` key is read by the build itself: ``network``, which is either
``bridge`` (the default — the service joins the compose network and publishes
the ports it wants reachable) or ``host`` (it shares the host's network
namespace, which is what a service needs to see broadcast traffic or reach
ports other software publishes on the machine). Your template has to render the
setting for it to mean anything, and ``osprey build`` refuses a service that
declares ``network: host`` whose render does not carry it. See
:ref:`deployment-network-attachment` for what host mode changes and for
``dispatch.network``, the single knob that covers the event dispatcher and its
workers.

.. _profile-graph-mode:

Graph-mode channel finding
==========================

``channel_finder_mode: graph`` points the channel finder at the deployment's
graph store: the agent searches the facility knowledge graph for channels
instead of reading a channel database. The store *is* the database, so the
profile ships no channel-database inputs and pins no ``tier`` — what it does
need is a ``services.graphdb`` block, and the paradigm works with either shape
that block comes in.

A deployment that runs its own store already has one. The ``control_assistant``
app template renders ``graphdb`` into ``services`` and ``deployed_services``,
and ``osprey up`` starts, bootstraps and seeds it — see
:doc:`/how-to/deploy-project/index`:

.. code-block:: yaml

   name: control-room
   app_template: control_assistant
   provider: anthropic
   channel_finder_mode: graph     # no `tier` — graph has no tiered artifacts

To search a store the facility already runs, name it instead of deploying one.
That takes three things: an explicit ``uri`` on the block, the store's password
in the project ``.env`` as ``GRAPHDB_PASSWORD`` (nothing is minted for a store
this deployment does not own), and ``graphdb`` off the ``deployed_services``
list — otherwise the build still stands up a local Neo4j nobody queries.
``deployed_services`` is a list, so the override replaces it whole; write out
the services you *do* want rather than the one you are removing.

.. code-block:: yaml

   channel_finder_mode: graph
   config:
     services.graphdb.uri: bolt://graph.facility.org:7687
     services.graphdb.username: neo4j    # default `neo4j`
     # Whole-list replacement: the template's default is
     # [postgresql, openobserve, qmd, graphdb].
     deployed_services: [postgresql, openobserve, qmd]

An external store has to already hold a corpus for the mode to answer anything:
generate one with ``osprey knowledge build-ttl`` and load it with ``osprey
knowledge seed-graph`` (see
:doc:`/how-to/facility-knowledge/use-facility-graph`). A store this deployment
runs gets that during ``osprey up``; a store it only connects to does not.

A profile whose app template carries no ``services.graphdb`` block at all — the
channel-finder app template carries none — is refused at build time, naming the
missing block, rather than rendering a channel finder with nothing to read. An
attached project (``deploy_services: false``) is refused the same way unless it
names an external store, because it renders ``services: {}`` whatever its app
template says. For what the mode changes about the agent's answers, see
:doc:`/how-to/use-channel-finder`; for the corpus behind them,
:doc:`/how-to/facility-knowledge/use-facility-graph`.

.. _profile-va-archiver:

The ``va_archiver`` block
=========================

A deployment that serves simulated channels still needs somewhere to keep what
those channels did. Declaring ``va_archiver:`` is what gives it one: the build
adds a MongoDB store and a recorder to the service stack, ``osprey up``
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
       derives no check (see
       :doc:`/how-to/health-and-monitoring/configure-health-checks`).
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
   :doc:`/how-to/control-systems/use-virtual-accelerator`.


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

The ``env`` section declares what the deployment needs. ``required`` documents
variables the operator must supply; secrets live in the repository's ``.env``
and never in the profile (see :ref:`profile-secrets`).

.. code-block:: yaml

   env:
     required:
       - API_KEY
       - DB_HOST
     defaults:
       LOG_LEVEL: info

Both lists are rendered into the repository's ``.env.example``, so an operator
opening that file sees them alongside every other variable. Required names must
match ``^[A-Z_][A-Z0-9_]*$``.

``defaults`` values are additionally seeded into the repository's ``.env`` by
``osprey init``, under their own section banner, so a deployment created from
the profile starts with them in force. Seeding is append-only: a value already
in the file — set by the operator, or minted by a deploy — always wins, and
later ``init`` runs never rewrite one. Declare a default only for a value the
profile's author can honestly choose for every deployment (the
``control-assistant`` preset's demo login passwords, say); values a *site*
should share across hosts belong in ``.env.shared``, which is committed with
the repository and read by every host (see :ref:`deployment-env-chain`).


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




bluesky
=======

The ``bluesky:`` section configures the Bluesky stack a deployment brings up —
the bridge, the queue server, the BLUESKY panel and, optionally, the Tiled data
store. It accepts exactly five keys; a misspelled or unknown key **fails the
build** and prints the valid set:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - What it does
   * - ``port``
     - The bridge's port (default 8090).
   * - ``tiled_enabled``
     - Deploy the Tiled data store alongside the stack.
   * - ``tiled_port``
     - Tiled's port (default 8091).
   * - ``plan_dir``
     - A directory of your facility's own plans — see
       :doc:`/how-to/bluesky/write-plans`.
   * - ``excluded_plans``
     - Plans to remove from the catalog entirely, e.g. ``[orm]``.

Whether a deployment can execute plans at all is not set here: it follows from
the control system the deployment runs. See :doc:`/how-to/bluesky/queue` for
what the stack looks like when it is up.
