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
   * - ``data``
     - string
     - *required*
     - Facility data tree, relative to the profile directory (``data`` in a
       materialized profile). It is the only source of the project's ``data/``
       tree, so a profile that names none is refused. May resolve outside the
       profile directory; only existence and shape are checked
       (see :ref:`profile-self-contained`).
   * - ``provider``
     - string
     - *required*
     - LLM provider. Names an entry in ``providers.yml``, the catalog beside
       ``profile.yml`` (see :ref:`profile-provider-catalog`). A name that file
       does not declare is refused, and the message lists the names it does.
       The build aborts if none is set.
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
   * - ``web_panels``
     - list
     - ``[]``
     - Panel tabs the web workspace offers beside the terminal. A framework
       panel is named by its id; your own is a list entry plus its address
       under ``config:`` (``web.panels.<id>.url`` and its siblings).
   * - ``default_panel``
     - string
     - ``None``
     - Panel id the web terminal opens on. Must be a built-in, an entry in
       ``web_panels``, or a custom panel backed by a ``web.panels.<id>.url``
       setting. Renders ``web.default_panel``.
   * - ``panel_presets``
     - mapping
     - ``{}``
     - Named web-terminal layouts, as label to list of panel ids. Renders
       ``web.presets``.
   * - ``config``
     - mapping
     - ``{}``
     - Dot-notation settings for the generated ``config.yml``. A materialized
       profile carries the whole block its preset ships, so this is where the
       deployment's declarative configuration is read and edited. A handful of
       keys are the build's to write rather than yours; spelling one here is
       refused (see :ref:`profile-derived-keys`).
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
   * - ``virtual_accelerator``
     - mapping
     - absent
     - Declares a simulated machine the deployment runs — and, optionally, a
       second copy of it standing in for the live target
       (see :ref:`profile-virtual-accelerator`).
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
     - Which preset this profile was materialized from, that preset's hash, and
       the hash of the ``providers.yml`` beside it. Written by the
       materialization; ``osprey validate`` compares the profile with that
       preset and refuses unmarked *structural* differences, while a key both
       carry with different values is only reported (see
       :ref:`profile-preset-drift`). The one key you may add is
       ``deviation_marker``, the tag of the ``# <TAG>: <why>`` comment that
       marks a difference as deliberate (default ``DEVIATION``).


Configuration overrides
=======================

The ``config:`` section uses **dot notation** to set any key in the generated
``config.yml``. Nothing is inherited from underneath it: a profile
``osprey init`` materialized carries every key its preset configures, each on
the line that documents it, so reading the block is reading the deployment's
configuration. ``osprey config --defaults`` prints the wider ledger — every key
the framework reads and what it falls back to when no line spells it.

.. warning::

   Always write overrides as **dotted keys**, one per line — never as nested
   YAML. A nested block counts as *one* override whose value replaces the entire
   subtree. ``config: {claude_code: {model: opus}}`` wipes out everything else
   under ``claude_code`` (servers, permissions, …), silently. The dotted form
   ``claude_code.model: opus`` changes just that setting.

.. code-block:: yaml

   # The channel-finder paradigm is a top-level field, not a `config:` key: the
   # build renders `channel_finder.pipeline_mode` and the per-mode pipelines
   # from it, and spelling either under `config:` is refused.
   channel_finder_mode: middle_layer

   config:
     # Control system: which backend the deployment talks to. Required — see
     # the posture floor below.
     control_system.type: live_standin
     # The write posture every connector type inherits when it says nothing
     # itself. Only a literal `true` arms writes, at either level.
     control_system.writes_enabled: true
     # Limits checking. This pair is the deployment's, and every type
     # inherits it ...
     control_system.limits_checking.enabled: true
     control_system.limits_checking.allow_unlisted_channels: false
     # ... while a per-type block replaces it whole for one type, and does not
     # fall back to the keys above. Both settings have to be stated: one alone
     # is refused by `osprey build` and `osprey validate`.
     control_system.connector.virtual_accelerator.limits_checking.enabled: true
     control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels: true

     # Archiver: where history is read from. Required alongside a control
     # system.
     archiver.type: mongodb_archiver

     # Set your real facility zone: it governs how the agent reads operator
     # times (parsed as facility-local) and renders every timestamp — not
     # just a display label.
     system.timezone: America/Los_Angeles

     # Approval policy
     approval.enabled: true
     approval.default_policy: always

That is the shape ``osprey init --preset control-assistant`` writes, with the
timezone changed. Pointing the same deployment at a real machine
(``control_system.type: epics``) is a larger edit than the one line, because
two things the preset ships are scoped to the simulated baseline: the
``va_archiver:`` block records a machine the deployment would no longer be
baselined on, and any profile-level ``control_system.connector.<type>.writes_enabled:
true`` has to agree with the read-only personas that inherit it. ``osprey
build`` refuses each in turn rather than rendering it. See
:doc:`/how-to/control-systems/use-virtual-accelerator`.

.. note::

   **Connector types whose name contains a dot** are the one place the flat form
   does not work. A custom connector is named by its module path, and the build
   splits only the *first* key of each ``config:`` entry on dots — so
   ``control_system.connector.mypkg.MoatConnector.limits_checking.enabled``
   renders the type's module path as two nested keys, ``mypkg`` and
   ``MoatConnector``, instead of the one key the connector is actually called.
   ``osprey build`` and ``osprey validate`` refuse that entry by name rather
   than letting it render somewhere nothing reads. Write such a type as its own
   map key under a dotted prefix, which is the one spelling that puts the block
   where the connector looks for it:

   .. code-block:: yaml

      config:
        control_system.connector:
          mypkg.MoatConnector:
            limits_checking:
              enabled: true
              allow_unlisted_channels: false

   **That entry replaces the whole rendered connector section.** ``connector``
   is the last key of the dotted prefix, and a leaf is assigned verbatim — so
   the mapping has to carry every connector block the deployment needs, not only
   the custom type's. Copy the ``mock``, ``virtual_accelerator`` and ``epics``
   blocks, with their gateways, ports and probe channels, out of the
   ``control_system.connector.*`` keys this profile already spells under
   ``config:``, or from a prior render's ``config.yml``.
   Nothing outside ``connector`` is disturbed, and nothing refuses a mapping
   that leaves a block out: the deployment simply comes up without the addresses
   that block held.

   What is never right is a bare ``control_system:`` mapping beside flat
   ``control_system.*`` keys — the bare top-level key is the only spelling that
   does not merge, since deeper dotted prefixes are walked into rather than
   assigned over. It replaces the whole rendered section, so which of the two
   reaches ``config.yml`` depends on key order. ``osprey build`` and ``osprey
   validate`` refuse that pair by name too. Built-in types have no dots and take
   the flat form.


.. _profile-derived-keys:

Keys the build renders
======================

A few config keys are not yours to state. The build writes them from the
project layout, the port layout, ``providers.yml``, or a top-level profile
field — so a ``config:`` line for one of them is a second home for one fact,
and the render would silently win. So ``osprey build`` and ``osprey validate``
refuse that line by name and say what supplies the value instead:

.. code-block:: text

   config: channel_finder.pipeline_mode is rendered by the build; the top-level
   `channel_finder_mode:` field sets it. Remove it from profile.yml.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Rendered key
     - Set it here instead
   * - ``claude_code.provider``, ``logbook.composition.provider``,
       ``ariel.enhancement_modules.semantic_processor.provider``
     - the top-level ``provider:`` field
   * - ``claude_code.default_model``,
       ``ariel.enhancement_modules.semantic_processor.model.model_id``
     - the top-level ``model:`` field
   * - ``channel_finder.pipeline_mode``, ``channel_finder.pipelines``
     - the top-level ``channel_finder_mode:`` field
   * - ``web.default_panel``
     - the top-level ``default_panel:`` field
   * - ``web.presets``
     - the top-level ``panel_presets:`` field
   * - ``api.providers``
     - ``providers.yml`` beside the profile (:ref:`profile-provider-catalog`)
   * - ``project_name``, ``project_root``, ``build_dir``, ``file_paths``,
       ``agent_data.base_dir``
     - nothing — the build takes them from the repository it renders into
   * - ``execution.environment.python`` / ``.packages`` / ``.inherit_exclude``
     - the ``environment:`` block (:ref:`profile-environment`)
   * - ``artifact_server.port``
     - ``config: deployment.port_base``, which moves the whole port block

Each entry claims every key beneath it, in every spelling: a dotted key, a
dotted prefix over a mapping, a fully nested block, or any mix reaches the same
rendered leaf and is refused the same way.

``web.panels.<id>.enabled`` is the one exception. It is derived from
``web_panels:`` too, but a ``config:`` line that *agrees* with the selection is
accepted; only one that contradicts it is refused.

.. _profile-provider-catalog:

The provider catalog — providers.yml
====================================

``providers.yml`` sits beside ``profile.yml`` and lists every model provider
the deployment can name. ``osprey init`` writes it; ``osprey build`` renders the
whole file into ``api.providers`` in ``build/config.yml``; the profile's
top-level ``provider:`` field picks one entry by name.

.. code-block:: yaml

   providers:
     my-gateway:
       api_key: ${MY_GATEWAY_API_KEY}
       base_url: https://my-gateway.example.org/v1
       models:
         haiku: claude-haiku-4-5
         sonnet: claude-sonnet-4-6
         opus: claude-opus-4-6

``base_url`` is required. ``api_key`` is optional and is normally an
``${ENV_VAR}`` reference resolved at run time from the repository's ``.env`` —
keys are never written into this file. ``models`` maps the ``haiku`` /
``sonnet`` / ``opus`` tiers onto the provider's own model IDs, so ``model:
sonnet`` means something whichever provider is selected. ``api_protocol:
anthropic`` marks an Anthropic-native endpoint rather than an
OpenAI-compatible one.

Three rules follow from the catalog being the one home for these facts:

- A ``config: api.providers.*`` key is refused, naming ``providers.yml`` as the
  place to declare the provider.
- ``provider:`` naming an entry the file does not declare is refused, and the
  message lists the names it does declare.
- A catalog that no longer matches the one OSPREY ships is reported as a note,
  never a refusal — the comparison is with the catalog this OSPREY release
  ships, not with the hash recorded under ``provenance:``.
  ``osprey profile expand --providers`` refreshes the entries OSPREY ships and
  keeps the ones you added.

.. _profile-posture-floor:

The posture floor
=================

Six config keys have no safe unstated answer, so ``osprey build`` refuses a
deployment that leaves one of them silent rather than letting a reader's
fallback decide. The refusal names the dotted key and ``profile.yml``. The gate
reads the *rendered* config, so a key a preset's ``config:`` block spelled, an
injector wrote, or a persona inherited counts as stated.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Key
     - Required when
   * - ``control_system.type``
     - the ``controls`` MCP server resolves enabled — without a type the
       connector factory picks the backend the deployment talks to
   * - ``archiver.type``
     - the ``controls`` MCP server resolves enabled — without a type the
       archiver factory picks the history the deployment reports
   * - ``approval.enabled``
     - the ``approval`` hook is in the profile's ``hooks:`` list; this key is
       whether it prompts at all
   * - ``approval.default_policy``
     - the ``approval`` hook is in the profile's ``hooks:`` list; this key is
       what it does with a tool nothing named
   * - ``claude_code.telemetry.enabled``
     - always
   * - ``hooks.debug``
     - always

A standalone that switches the controls server off — the ARIEL and
channel-finder presets do — is asked for neither control-system key: it has no
control system to state a type for.

.. _profile-preset-drift:

Preset drift, and ``osprey profile expand``
===========================================

A profile ``osprey init`` wrote records the preset it came from under
``provenance:``. ``osprey validate`` compares the two and splits what it finds:

- A **structural** difference — a key, block or list member one document has
  and the other has not — is refused, with both ``file:line`` references,
  unless a ``# DEVIATION: <why>`` comment within three lines above the line
  claims it (for a line the profile does not have, a comment anywhere naming
  the key or member). ``--drift=warn`` reports these and passes.
- A **value** difference on a key both documents carry is only reported. Your
  profile is the source of truth, and a preset that changes a value in a newer
  OSPREY release should not fail your build.

So the external-store recipe below, which takes ``graphdb`` out of
``deployed_services`` and adds ``services.graphdb.uri`` and ``.username``, is
three structural differences and wants marker comments; changing ``web.theme``
from the preset's ``light`` to ``dark`` is a value difference and wants
nothing.

A profile written before the ``config:`` block carried everything spells only
what its facility changed, and still names a packaged app template at its top
level. Such a profile is refused, naming that key and the verb that fixes it:

.. code-block:: bash

   osprey profile expand                          # preset from provenance:
   osprey profile expand --from control-assistant # or name it
   osprey profile expand --providers              # refresh providers.yml too

``expand`` writes every key the preset documents and the profile lacks into the
profile's own ``config:`` block, each under the comment the preset documents it
with, and drops that retired key. It only adds — nothing
already in the file is touched — so running it twice leaves the profile exactly
as the first run left it. See :ref:`cli-profile-expand` for the two classes of
key it deliberately leaves out and for what it prints.

.. _profile-mcp-servers:

MCP server injection
====================

The top-level ``mcp_servers:`` key declares the MCP servers a build wires into
the agent. The build records each entry in the rendered ``config.yml`` (under
``claude_code.servers``) and renders it from there into ``.mcp.json`` (server
configuration) and ``.claude/settings.json`` (tool permissions) — so a later
``osprey build`` re-renders them instead of losing them. For the procedure, see
:doc:`/how-to/agent-interfaces/add-mcp-server`.

.. code-block:: yaml

   mcp_servers:
     my_server:
       command: "{current_python_env}"
       args: ["-m", "my_server"]
       env:
         CONFIG: "{project_root}/build/config.yml"
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
``transport`` (launching via ``command`` *is* the transport). A ``port:`` may
stand in for a whole ``url``: for an HTTP service the deployment publishes on
one port, ``port: 8008`` derives ``http://localhost:8008/mcp``. It is
mutually exclusive with ``command`` for the same reason ``url`` is, and it
cannot stand in for an ``sse`` server's ``url``: ``transport: sse`` always
requires an explicit ``url``, whether or not a ``port`` is given, because an
event stream does not live at the derived ``/mcp`` path. The derived block the
build records alongside a ``port:`` names the container URL under the server's
own key — so name the compose service after the ``mcp_servers:`` key or that
URL points at no host.

**Placeholders:** ``{project_root}`` resolves at build time to the absolute
project path, and ``{current_python_env}`` to the interpreter the framework's
own MCP servers run under (the project venv under the default
``python_env: project``; see the key table above), which is what a Python
server shipped with the profile should launch under too. ``${ENV_VAR}`` is
preserved for the container or shell to resolve at runtime.

**Permission wiring:** for a server named ``my_server`` with
``allow: ["safe_tool"]``, the build adds ``mcp__my_server__safe_tool`` to the
allow list.

Shipping the server's code
--------------------------

Put the package in the profile's ``mcp_servers/`` directory — one directory per
server. The build copies it to ``build/_mcp_servers/`` in the project, so the
launch command finds it:

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
       command: "{current_python_env}"
       args: ["-m", "phoebus"]
       env:
         OSPREY_CONFIG: "{project_root}/build/config.yml"
         PYTHONPATH: "{project_root}/build/_mcp_servers"
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

The third source is the one to know about. A profile that leaves **no** control
target armed does not merely stop the write path at run time; it also puts the
framework servers' write tools into the rendered deny list. Those entries are
generated, not authored, and no ``remove_deny`` reaches them. (Tools a profile
lists under ``control_system.write_tools`` are a different mechanism: they
never reach ``permissions.deny``, in any render, and are refused by the
writes-check hook instead.) A profile that arms nothing and also writes
``remove_deny: ["mcp__controls__channel_write"]`` still renders that deny —
which is the point. A read-only posture that a later edit could quietly lift
would not be a posture at all.

Arming *some* targets and not others renders differently, and the difference is
worth knowing before you write such a profile. ``settings.json`` is rendered
once, before any session has picked a target, so a tool that is legal on the
simulator and refused on the machine cannot be denied there — and it cannot be
left in ``ask`` either, or an operator would be prompted to approve a write the
target's posture forbids. Such a profile therefore renders **neither**: the
gated tools leave both lists, and the boundary is carried per call by the
safety hook (which reads the session's active target) and by the connector
behind it. The static deny is the stronger of the two, so reach for it — a
profile that arms nothing — whenever the requirement is "this tier can never
move anything".

.. admonition:: Permission lists grow across ``extends``; taking an entry out is explicit
   :class: important

   The permission lists under ``config:`` — ``deny``, ``ask`` and their
   ``remove_*`` companions — **union** with the ones they inherit. A child
   profile or a persona delta can add to an inherited permission list; to
   take an entry out it has to say so, with ``remove_deny`` / ``remove_ask``
   (which match by substring) or with ``exclude: config:`` naming the exact
   entry (see :ref:`profile-exclude`).

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
   build`` and ``osprey up`` alike, wherever the deployment has a login page
   at all — with ``auth.method`` at ``token`` (the default) or ``none`` there
   is no wall to be exempt from, and the same exposure is reported as an
   advisory instead. On a profile that
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

A second ``config`` key the build reads is ``env``: a list of host environment
variable NAMES the service passes through to its container, in the order you
write them.

.. code-block:: yaml

   services:
     typesense:
       template: services/typesense
       config:
         env: [TYPESENSE_TELEMETRY, HTTPS_PROXY]

Names only, never values — each becomes a ``NAME: ${NAME}`` the deploy env
chain fills in, so rotating a value is an edit to ``.env`` and a restart. A name
nothing sets arrives as an empty variable rather than an absent one.

The event dispatcher and its workers take the same axis as ``dispatch.env``,
one list for both halves, because the build writes both of their service blocks
itself:

.. code-block:: yaml

   dispatch:
     triggers: my_triggers.yml
     env: [EPICS_CA_ADDR_LIST, EPICS_CA_NAME_SERVERS]

Writing ``env:`` on either half's own service block is refused for the same
reason ``network:`` there is: the build would overwrite it.

.. _profile-graph-mode:

Graph-mode channel finding
==========================

``channel_finder_mode: graph`` points the channel finder at the deployment's
graph store: the agent searches the facility knowledge graph for channels
instead of reading a channel database. The store *is* the database, so the
profile ships no channel-database inputs and pins no ``tier`` — what it does
need is a ``services.graphdb`` block, and the paradigm works with either shape
that block comes in.

A deployment that runs its own store already has one.
``osprey init --preset control-assistant`` writes the ``services.graphdb.*``
keys and the ``graphdb`` entry in ``deployed_services`` into the profile's
``config:`` block, and ``osprey up`` starts, bootstraps and seeds it — see
:doc:`/how-to/deploy-project/index`. That profile reads, in outline:

.. code-block:: yaml

   name: control-room
   provider: anthropic
   channel_finder_mode: graph     # no `tier` — graph has no tiered artifacts
   data: data
   config:
     services.graphdb.path: ./services/graphdb
     services.graphdb.image: neo4j:5.26-community
     services.graphdb.ttl_path: ./data/demo_machine.ttl
     # ... the JVM and query-bound keys the preset also ships
     deployed_services: [postgresql, openobserve, qmd, graphdb]

The mode reads the same ``services.graphdb`` block when the store is one the
facility already runs and this deployment only connects to. The keys that
express that are an explicit ``uri``, ``username`` (default ``neo4j``), and a
``deployed_services`` list without ``graphdb`` in it. An edit replaces that list
whole, so write out the services you *do* want rather than the one you are
taking away:

.. code-block:: yaml

   channel_finder_mode: graph
   config:
     # DEVIATION: the graph store is the facility's, not this deployment's
     services.graphdb.uri: bolt://graph.facility.org:7687
     services.graphdb.username: neo4j
     # DEVIATION: graphdb is the facility's store, so deployed_services drops it
     deployed_services: [postgresql, openobserve, qmd]

Those are structural differences from the preset — two added keys and a dropped
list member — which is why the marker comments are there. Without them
``osprey validate`` refuses the profile (:ref:`profile-preset-drift`).

The procedure that goes with those keys — the password to place, what is minted
and seeded on each path, and how to load a corpus into a store OSPREY does not
run — is in
:ref:`Pointing at a store the facility runs <graph-external-store>`. A store
this deployment runs is seeded during ``osprey up``; a store it only connects
to is not.

Removing the store means deleting those keys. A whole-block
``config: services.graphdb: {}`` or a bare ``services.graphdb:`` with no value
is refused by name: an empty block reads as "a store at the defaults" wherever
the block is resolved, and a null one is what every resolver downstream trips
over. Neither removes anything, because nothing injects a store to subtract.

A profile that renders no ``services.graphdb`` block at all — the
channel-finder-standalone preset ships none — is refused at build time, naming
the missing block, rather than rendering a channel finder with nothing to read.
An attached project (``deploy_services: false``) is refused the same way unless
it names an external store, because it renders ``services: {}`` whatever its
own ``config:`` block says. For what the mode changes about the agent's
answers, see :doc:`/how-to/use-channel-finder`; for the corpus behind them,
:doc:`/how-to/facility-knowledge/use-facility-graph`.

The qmd sidecar behind hybrid logbook search is guarded the same way. The
``control-assistant`` and ``ariel-standalone`` presets switch
``ariel.search_modules.hybrid`` on and deploy the ``services.qmd`` sidecar that
answers it, but an attached project renders ``services: {}`` — so on its own it
would keep the mode with nothing behind it, and every logbook query would fail
with *no qmd sidecar is configured*. ``osprey build`` refuses that instead of
rendering it. A deploying profile is held to the same rule from the other side:
a client left switched on for a service the profile no longer deploys — the
bluesky MCP server after the ``bluesky:`` block was removed — is refused naming
the service missing from ``deployed_services``, unless the profile names one
this deployment does not run (an external store's ``uri``, an ARIEL database
DSN).

An attached project rarely has to say anything, because **the build tells it
where the host's services are**. Every client-facing fact — the sidecar's port,
the graph store's bolt port, the Postgres the logbook lives in, the telemetry
store's port, the Bluesky bridge, the EVENTS and BLUESKY tab URLs — is copied
from the hosting deployment's own render into each persona built beside it, so
a service moved on the hosting profile moves every persona with it, and the
shipped ``control-assistant-*`` presets pin none of them. A persona profile
that spells one of these keys with a *different* value is refused — the two
copies would dial different places, and the build names both. (A persona
inherits the hosting profile's ``config:`` keys, so a port moved there is
spelled in every persona as the host's own value, and agrees.) A persona
built *alone* (``osprey init --preset control-assistant-logbook`` in a
repo with no hosting deployment) is told what its own preset deploys at the
shipped defaults instead, and there its ``config:`` is where a host that
differs is named:

.. code-block:: yaml

   config:
     services.qmd.port: 9180        # the sidecar of the deployment this build shares a host with
     # or: ariel.search_modules.hybrid.enabled: false

The same facts are checked again at run time: the ``reach`` category of
``osprey health`` (and of the system-health tab, inside each user's container)
resolves every live client's endpoint the way the client does and knocks on it
(:doc:`/how-to/health-and-monitoring/configure-health-checks`).

.. _profile-virtual-accelerator:

The ``virtual_accelerator`` block
=================================

Declaring ``virtual_accelerator:`` gives the deployment a simulated machine: a
soft-IOC container that speaks EPICS over a real Channel Access port, brought up
by ``osprey up`` and served from the lattice the build renders (see
:doc:`/how-to/control-systems/use-virtual-accelerator`).

.. code-block:: yaml

   virtual_accelerator:
     port: 5064
     live_standin: true

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Key
     - Default
     - Meaning
   * - ``port``
     - ``5064``
     - Channel Access port the simulator serves on. The connector block the
       build writes follows this value, so changing it moves both.
   * - ``live_standin``
     - absent
     - Deploy a **second** copy of the simulator as a third control target of
       its own, ``standin``. ``true`` puts it on the port layout's
       ``va_standin`` slot (``10090`` at the default base — see
       :ref:`reference-ports`); a number pins it somewhere else. Absent means
       one machine, as before.

The live stand-in
-----------------

``live_standin`` deploys a second simulator container on its own Channel Access
port and gives the deployment a third control target, ``standin``, dialled
through its own ``control_system.connector.live_standin`` block. Both containers
run one image over the same lattice and the same active scenarios; what differs
is a small fixed offset on the stand-in's BPM readouts, which is what lets you
tell the two apart by reading them.

The stand-in is a machine of its own, **not** a rewrite of ``live``. The build
never writes a key under ``control_system.connector.epics``, so ``live`` means
the gateways the facility authored on a stand-in deployment exactly as on one
without: a facility already pointed at its own control system can stand a
rehearsal up beside it. What the stand-in rehearses is the procedure, not the
risk — ``control_target_set standin`` moves a session onto it, and
``control_target_set live`` from there walks the real go-live path. See
:doc:`/how-to/control-systems/switch-control-target`.

**One fact, one home.** The key is where the stand-in is described, and the build
derives exactly seven keys from it, all under
``control_system.connector.live_standin``:

- the six ``gateways.*`` keys — ``address``, ``port`` and ``use_name_server``
  for each of the ``read_only`` and ``write_access`` roles, pointing both lanes
  at loopback on the stand-in's port over the Channel Access name server;
- ``probe_channel``, copied from the simulator's own, since the stand-in is the
  same soft IOC over the same machine model. A deployment whose simulator names
  none gets none here either.

A profile that spells any of the seven in its own ``config:`` block is refused by
name at build time rather than silently having the derived copy win: two homes
for one fact are free to disagree, and an address left in ``config:`` reads as
the endpoint the ``standin`` target dials while every session on it is somewhere
else. The refusal is scoped to those leaves — a persona's own
``control_system.connector.live_standin.writes_enabled`` says something the build
has no opinion about, and is yours to write.

**What the block does not decide.** Write posture, limits checking and the
operator acknowledgment are the profile's, on a stand-in deployment exactly as on
any other: they describe how the *deployment* is run, not where one of its
targets lives. In particular, a switch to ``standin`` (like one to ``live``)
requires the strict limits posture, so a profile that stands a stand-in up
normally writes the pair itself:

.. code-block:: yaml

   config:
     control_system.limits_checking.enabled: true
     control_system.limits_checking.allow_unlisted_channels: false

That pair is the deployment's, and the stand-in inherits it: the build writes no
``limits_checking`` block under ``control_system.connector.live_standin``, and a
profile should not either, since a permissive block there would make
``control_target_set standin`` refuse the very rehearsal the stand-in exists
for. A simulator beside it is where a per-type block belongs — see
:ref:`limits-checking-config`.

``control_system.target_switch.live_gateway_acknowledged`` stays the live
machine's alone — the stand-in's equivalent is the ``live_standin`` line itself.

**What the build refuses.** Beside the duplicate-key refusal above:

- a stand-in port that collides with ``virtual_accelerator.port``, with another
  port this profile spends, or with a hand-authored virtual-accelerator gateway
  port — the simulator and its stand-in are two endpoints, never one;
- a stand-in on a build with no built-in lattice behind it, because the shipped
  readout perturbation needs a model to displace and the IOC treats a
  perturbation it cannot apply as fatal at boot;
- ``control_system.type: live_standin`` on a profile that sets no
  ``virtual_accelerator.live_standin`` — a baseline naming a machine the
  deployment does not stand up;
- a ``va_archiver`` block beside a stand-in on a baseline that is neither
  simulated nor the stand-in, since the recorder would sample one machine into a
  store the deployment reads as another's.

On a laptop the second container is a real cost — the simulator image is
amd64-only, so Apple Silicon emulates both — and deleting the line is the
remedy; the how-to above says when that is worth it.

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
store. It accepts exactly eight keys; a misspelled or unknown key **fails the
build** and prints the valid set:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - What it does
   * - ``port``
     - The bridge's port. Defaults to its slot in the port layout, ``10080``
       at the default base (:ref:`reference-ports`); the second plan lane takes
       the next slot up.
   * - ``tiled_enabled``
     - Deploy the Tiled data store alongside the stack.
   * - ``tiled_port``
     - Tiled's port. Defaults to the layout's ``10070``.
   * - ``second_lane``
     - Run a second plan lane, so one deployment serves both its live
       machine and its virtual accelerator.
   * - ``plan_dir``
     - A directory of your facility's own plans — see
       :doc:`/how-to/bluesky/write-plans`.
   * - ``excluded_plans``
     - Plans to remove from the catalog entirely, e.g. ``[orm]``.
   * - ``devices_file``
     - The file listing the devices plans may drive or record
       (default ``data/bluesky_devices.yml``) — see
       :doc:`/how-to/bluesky/write-plans`.
   * - ``device_page_size``
     - How many devices the bridge lists at once (default 500). A larger set
       is served a page at a time and can be narrowed by an exact name
       prefix. The same number decides when a refusal for an unknown device
       stops listing every device it does know and gives a count instead.

Whether a deployment can execute plans at all is not set here: it follows from
the control system the deployment runs. See :doc:`/how-to/bluesky/queue` for
what the stack looks like when it is up.
