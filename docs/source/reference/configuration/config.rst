.. _reference-config:

Runtime Configuration — config.yml
==================================

``config.yml`` is the file an OSPREY deployment actually runs on. ``osprey
build`` renders it from the build profile, so the profile is where you edit a
setting and this file is where you look one up: :doc:`profile` describes the
authoring side, and this page catalogues what the rendered result means.

Three parts of that file are gathered here — the diagnostic suite (``health:``),
the browser UI's documentation and feedback settings (``web:``), and the
deployment keys that decide which container image each service runs and how
``${VAR}`` placeholders in the compose files are filled in. Settings that only
ever arrive from the environment are in :doc:`environment-variables`. A closing
note records the **protected set** — the files and keys no agent-side writer
may touch.

.. _config-health:

``health:`` — the diagnostic suite
----------------------------------

``osprey health`` always runs its built-in checks; a ``health:`` block adds a
facility's own checks and tunes the suite's timing. Everything in the block is
optional. For how to put these settings to work — the two recipes worth
starting from, cost classes, and the ``SYSTEM`` dashboard — see
:doc:`/how-to/health-and-monitoring/configure-health-checks`; for the shape of
the ``--json`` report, see :doc:`/reference/contracts/health-json`.

The ``health:`` block
~~~~~~~~~~~~~~~~~~~~~

All configuration lives under a top-level ``health:`` key. Every field is
optional; an absent ``health:`` block runs the built-in checks with their
default timing.

.. code-block:: yaml

   health:
     suite_timeout_s: 30          # poll-class wall-clock budget (default 30)
     on_demand_timeout_s: 120     # on_demand wall-clock budget (default: sum of budgets)
     interval_s: 300              # minimum server-side re-run interval

     plugins:
       - my_facility.health       # dotted module paths to plugin modules

     categories:
       beamline_services:         # a facility-defined category of probe checks
         checks:
           - name: archiver
             type: http
             url: http://archiver.example.com/healthz
           - name: bluesky_server
             type: mcp
             url: http://localhost:8931/mcp

       providers:                 # metadata-only override of a built-in category
         timeout_s: 15

Probe checks
~~~~~~~~~~~~

A **declarative category** is a named entry under ``health.categories`` with a
``checks:`` list. Each check names a probe ``type`` and its parameters; the
suite runs the checks and grades each result ``ok`` / ``warning`` / ``error`` /
``skip``. Six probe types ship:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - ``type``
     - Purpose and parameters
   * - ``http``
     - GET a URL and grade the response. ``url`` (required); ``expect_status``
       (default ``200``); ``warn_latency_ms`` / ``error_latency_ms`` (optional
       latency ceilings — over the warn ceiling is a ``warning``, over the error
       ceiling an ``error``).
   * - ``mcp``
     - Handshake an MCP server over streamable HTTP and list its tools. ``url``
       (required); ``expect_tools`` (optional list of tool names that must be
       present). With no ``expect_tools``, a server exposing zero tools is an
       ``error``.
   * - ``container``
     - Check a deployed container's state and healthcheck. ``container``
       (required; alias ``service``) — the container/service name, matched
       fuzzily against the running containers. Not-deployed or non-running is a
       ``warning``; **no container runtime installed is a** ``skip``.
   * - ``channel_read``
     - Read one control-system channel through the suite's connector and grade
       the value. ``address`` (required); ``expect`` (required exact value), or
       inclusive numeric bands ``ok_range: [lo, hi]`` and ``warn_range:
       [lo, hi]`` (outside the warn band is an ``error``, outside the ok band a
       ``warning``). With neither, a successful read is a liveness ``ok``.
   * - ``provider_canary``
     - Make a minimal connectivity call to a model provider. ``provider`` — the
       provider name to test (e.g. ``cborg``); ``api_key`` / ``base_url``
       (optional, ``${VAR}`` allowed; fall back to
       ``api.providers.<provider>``); ``model_id`` (optional). A canary never
       emits ``error`` — an unreachable provider is a ``warning``.
   * - ``archiver_freshness``
     - Verify the deployment's archiver is reachable **and actually
       accumulating data**: query the newest archived sample of a canary
       channel through the ``archiver:`` connector. ``channel`` (required);
       ``max_age_s`` (default 600) — a newest sample older than this is a
       ``warning``, as is an empty query window. An unreachable archiver, or an
       ``archiver_freshness`` check declared with no ``archiver:`` configured,
       is an ``error``. A reachable archiver UI does not prove data is flowing
       — this probe checks the data. A project that deploys its own archive can
       have this check **derived** rather than declared — see
       :doc:`/how-to/health-and-monitoring/configure-health-checks`.

.. note::

   For ``provider_canary``, ``name:`` and ``provider:`` are distinct: ``name``
   is the check's **result identity** (the row label in the report), while
   ``provider`` selects **which provider to test**. When ``provider`` is
   omitted, the probe falls back to ``name`` — so a check named after the
   provider works, but to test one provider under a different row label you must
   set both.

Every check also accepts the reserved keys ``name`` (required, unique within its
category), ``timeout_s``, ``timeout_status``, and ``requires:`` (below). Any
other key becomes a probe parameter.

Built-in service categories
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond the always-on framework checks, four built-in categories are
**presence-gated on their config blocks**: they contribute rows only when the
corresponding service is configured, so a minimal build shows no empty tiles.

- ``ariel`` — appears when a top-level ``ariel:`` block is configured. Probes
  the ARIEL interface's status endpoint and reports: reachability, logbook
  entry count, last ingestion time, and the registered search and enhancement
  modules. The interface sidecar runs with ``osprey web``, so a CLI-only run
  on a stopped stack reports the interface as unreachable (a ``warning``).
- ``channel_finder`` — appears when a top-level ``channel_finder:`` block is
  configured. Reports the active pipeline mode, verifies the pipeline's
  channel-database file exists (a configured-but-missing database is an
  ``error``), shows the database's age, and — for the ``middle_layer``
  pipeline — the channel count from the materialized DuckDB.

  The ``graph`` pipeline reads different rows, because it answers from the
  graph store rather than from a file on disk. The pipeline row reports it as
  store-backed, and in place of the database rows the category reports the
  store's **reachability** and its **resource count** — the same two readings
  the ``graphdb`` category takes, so a stopped store warns rather than
  failing the suite. A graph-mode build configures no ``database.path`` and is
  never warned about one missing: there is no channel-database file in that
  paradigm to look for (see :doc:`/how-to/use-channel-finder`). The ``graphdb`` tile is
  always alongside — graph mode needs that block — so these rows restate the
  same store readings in the channel finder's own terms: whether *channel
  search* has anything to answer from.
- ``graphdb`` — appears when a ``services.graphdb`` block is configured (see
  :doc:`/how-to/deploy-project/index`). Two rows: **connection**, which dials the store over
  bolt and reports the round-trip latency, and **resources**, the number of
  ``(:Resource)`` nodes in the graph — the nodes the TTL corpus imports.
  Bootstrapping a store creates neosemantics bookkeeping nodes whether or not a
  corpus was ever loaded, so counting ``(:Resource)`` specifically is what keeps
  an empty graph from reading as a populated one; a count of zero warns and
  names ``osprey knowledge seed-graph`` as the remedy. The agent's own graph
  tools report the same degraded states from the inside — see
  :doc:`/how-to/facility-knowledge/use-facility-graph`.
- ``reach`` — appears when this render has a client switched on for a shared
  service: hybrid logbook search and the OKF panel (the qmd sidecar), the
  graph MCP server (the graph store), the ARIEL panel and MCP server
  (Postgres), agent telemetry (the OpenObserve store), the Bluesky MCP server
  (the bridge), the EVENTS and BLUESKY tabs (their sidecars), and the
  virtual-accelerator connector. One row per service, and the address it
  knocks on is the one **the client itself resolves** from this config, in the
  process running the check — so the same category answers a different
  question on the host (what ``osprey`` commands reach) and inside a per-user
  web-terminal container, where the health MCP server runs it (what that
  user's agent reaches). A knock is a TCP connect; a closed port warns and
  names the config key the client resolved it from, and a client switched on
  with nothing to resolve warns with the key that would give it one — the
  same condition ``osprey build`` refuses (:doc:`/how-to/build-profiles`).

.. note::

   The ``graphdb`` category dials **bolt** — ``services.graphdb.port_host``, or
   an explicit ``services.graphdb.uri`` — never the ``http_port_host`` the Neo4j
   Browser and the container healthcheck use. That is the address its remedies
   name, so a probe that fails is pointing at the port the seeder also uses.

   Every row it produces is ``ok`` or ``warning``, never ``error``. A store that
   is stopped, unreachable, rejecting its credential, or simply unseeded is a
   service that is not running rather than a broken build, and the category says
   so in one line instead of failing the suite.

All three are ordinary categories: valid under ``--category``, tunable via a
metadata-only override, and rendered as dashboard tiles with no extra
configuration.

Auto-derived MCP server checks (``health.auto``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every MCP server your build wires up is already described in ``config.yml``
under ``claude_code.servers``. Rather than make you re-declare each one as an
``mcp`` probe, the framework reads those blocks and builds an ``mcp_servers``
category for you — one reachability check per server. It is on by default, so
you normally configure nothing.

.. code-block:: yaml

   health:
     auto:
       mcp:
         enabled: true        # default true — set false to drop the category
         url_key: host_url    # which URL to probe (see below)

A server is included when its ``claude_code.servers`` block carries a ``url`` and
a ``network`` block with a URL to reach it. Servers without those are skipped.

**Which URL is probed.** Each server block records two URLs: ``host_url`` (the
server seen from the machine, ``http://localhost:<port>/mcp``) and ``docker_url``
(the server seen from inside a container, ``http://<name>:<port>/mcp``). The
category picks one automatically:

- If you set ``url_key`` explicitly, that choice is always used.
- Otherwise the framework detects whether the health check is itself running
  inside a container — the runtime's own marker file, ``/.dockerenv`` under
  Docker or ``/run/.containerenv`` under Podman, or an ``OSPREY_IN_CONTAINER``
  environment variable you set yourself — and uses ``docker_url`` when
  containerized, ``host_url`` on a plain host. Nothing in the shipped
  deployment sets that variable; set ``url_key`` explicitly when the automatic
  answer is wrong for your network layout (a host-networked container is in a
  container but cannot resolve compose service names).

**Expected tools.** If a server block declares ``permissions`` (its ``allow``
and ``ask`` tool lists), the derived check also confirms the server actually
exposes those tools, not just that it answers — a reachable server missing an
expected tool is graded an ``error``. A server with no declared permissions gets
a plain reachability check.

.. warning::

   ``docker_url`` assumes the container's hostname matches the server's key in
   ``config.yml`` — a server named ``matlab`` is probed at ``http://matlab:…``.
   If your compose service is named differently, the probe points at a host that
   does not exist. Either rename the service to match the key, or set
   ``url_key: host_url`` to probe the localhost URL instead.

Timeouts
~~~~~~~~

``timeout_s`` bounds a single check. Omit it and the probe's per-type default
applies:

.. list-table::
   :header-rows: 1
   :widths: 40 25

   * - Check
     - Default ``timeout_s``
   * - ``http``
     - 5
   * - ``mcp``
     - 10
   * - ``container``
     - 10
   * - ``channel_read``
     - 5
   * - ``provider_canary``
     - 5
   * - ``archiver_freshness``
     - 10
   * - callable category (poll)
     - ``suite_timeout_s``
   * - callable category (on_demand)
     - 60

``timeout_status`` sets the status emitted when a check's own timeout fires —
``error`` (the default) or ``warning``. Use ``warning`` for a dependency whose
unresponsiveness should be treated as non-fatal:

.. code-block:: yaml

   health:
     categories:
       beamline_services:
         checks:
           - name: archiver
             type: http
             url: http://archiver.example.com/healthz
             timeout_status: warning   # a slow archiver warns, never errors

.. warning::

   ``timeout_status: warning`` composes literally with ``requires:`` (below).
   A ``requires:`` dependency *passes* when its status is ``ok`` **or**
   ``warning`` — so a dependency that times out under ``timeout_status:
   warning`` still counts as passed, and its dependents still run. Setting
   ``timeout_status: warning`` on a check that *gates* others is therefore an
   explicit opt-in to "an unresponsive dependency is non-fatal"; leave it at the
   default ``error`` if a timed-out gate should skip everything downstream.

Dependencies (``requires:``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A check may declare ``requires:`` — a list of *earlier* checks in the **same
category** that must pass before it runs. A dependency passes when its status is
``ok`` or ``warning``; if any dependency does not pass, the dependent is emitted
as ``skip`` without running, and that ``skip`` in turn fails *its* dependents
(the cascade). Independent checks in a category still run concurrently — only a
genuine dependency chain serializes.

.. code-block:: yaml

   health:
     categories:
       beamline_services:
         checks:
           - name: gateway
             type: http
             url: http://gateway.example.com/healthz
           - name: archiver
             type: http
             url: http://archiver.example.com/healthz
             requires: [gateway]   # skipped if the gateway check did not pass

A dependency must reference a check declared earlier in the list; a forward
reference, a self-reference, an unknown name, or a duplicate check name is a
configuration error at load time.

Category metadata overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``health.categories.<name>`` entry with **no** ``checks:`` list is a
metadata-only override. It may set ``cost`` (``poll`` / ``on_demand``) and/or
``timeout_s`` for a category defined elsewhere — a built-in (core) category or a
plugin category — without redefining it:

.. code-block:: yaml

   health:
     categories:
       providers:
         timeout_s: 15        # give the built-in providers category a longer budget
       model_chat:
         cost: poll           # run model_chat on every health check (use with care)

The reverse is rejected: a ``checks:`` list under a built-in category name is a
load-time error ("cannot redefine built-in category") — use metadata-only keys
to adjust a built-in, and a new category name for your own probe checks.

Suite timing
~~~~~~~~~~~~

Three scalar settings tune the suite as a whole:

- ``suite_timeout_s`` (default 30) — the wall-clock budget bounding all
  poll-class categories collectively. It is also the default budget for a
  poll-class callable category.
- ``on_demand_timeout_s`` — the wall-clock budget bounding all on_demand
  categories collectively (only relevant under ``--full``). When omitted it
  defaults to the sum of the selected on_demand categories' budgets.
- ``interval_s`` — the minimum interval between server-side re-runs. When
  omitted it derives as ``max(60, 2 × suite_timeout_s)``; an explicit value must
  be greater than ``suite_timeout_s`` or the config is rejected. This value is
  validated but not yet enforced by ``osprey health`` itself.

At a cost-class deadline, unfinished checks are not dropped — every configured
check still produces a row (an eligible pending check becomes an ``error``
"suite deadline exceeded"; a pending check whose dependency failed becomes a
``skip``), so the report always accounts for every declared check.

.. _config-web:

``web:`` — documentation, feedback and the chat pool
----------------------------------------------------

The top-level ``web:`` section configures the browser UI the Web Terminal
renders — not the terminal process itself, which has its own ``web_terminal:``
section. The keys below aim the rail's two utility controls, bound the feedback
store, name the deployment, and size the Simple-mode operator-chat pool.

.. _feedback-configuration:

Documentation and feedback keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - Key
     - Default
     - What it does
   * - ``web.docs_url``
     - ``https://als-apg.github.io/osprey``
     - Where the **Documentation** control — in the rail and in the status bar
       — points. Set it to your own hosted copy of the docs.
   * - ``web.feedback.github_repo``
     - ``als-apg/osprey``
     - ``owner/repo`` the feedback dialog's GitHub channel opens a prefilled
       new issue against.
   * - ``web.feedback.email``
     - ``thellert@lbl.gov``
     - Recipient of the prefilled mail draft the dialog's Email channel opens.
   * - ``web.feedback.max_store_bytes``
     - ``268435456`` (256 MB)
     - Ceiling on the on-disk feedback store. Over it, the oldest saved session
       contexts are dropped; the submissions themselves are always kept.
   * - ``web.app_name``
     - unset (no badge)
     - Optional name badge in the terminal header, so otherwise-identical
       deployments are told apart. The ``OSPREY_WEB_APP_NAME`` environment
       variable outranks it, which is what lets several containers sharing one
       baked config image each carry their own name.
   * - ``web.chat_turn_timeout_s``
     - ``600``
     - Ceiling in seconds on a single Simple-mode chat turn.
   * - ``web.chat_idle_timeout_s``
     - ``1800``
     - How long an operator's chat session may sit idle before it is reaped.
   * - ``web.chat_max_sessions``
     - ``5``
     - Cap on concurrent operator-chat sessions.

.. code-block:: yaml

   web:
     docs_url: https://docs.example-facility.org/osprey
     feedback:
       github_repo: example-facility/controls
       email: controls-support@example.org
       max_store_bytes: 268435456

The last four keys are unset in every shipped template: they are read straight
from ``config.yml`` when present and fall back to the defaults above when not.

Blank, absent, and written-with-no-value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three ways of writing one of the three **string** keys mean three different
things:

- **Leave the key out** and the deployment uses the shipped default above.
- **Set it to an explicitly blank value** (``docs_url: ""``) and the deployment
  declares it has no such target: the Documentation link is not rendered at all,
  or the matching feedback channel is refused with an explanation rather than
  aimed at the upstream maintainers. This is the air-gapped posture — blanking
  ``web.docs_url`` is how you avoid shipping a link that opens a dead tab, and
  blanking ``web.feedback.github_repo`` retires the GitHub channel instead of
  aiming reports at the upstream maintainers' tracker.
- **Write the key with no value at all** (``docs_url:`` and nothing after it)
  and it reads as *absent*, not blank — "I have not decided yet" rather than
  "there is none" — so you get the default. Write ``""`` when you mean none.

``max_store_bytes`` takes a positive byte count; anything else — blank
included — is reported in the log and the default is used. The feedback
dialog's Local channel is always available and needs no configuration at all.

A build profile overrides any of these keys from its ``config:`` block in the
dotted form, e.g. ``web.feedback.max_store_bytes: 536870912``.

.. _config-deployment:

Deployment — service images and ``${VAR}`` interpolation
--------------------------------------------------------

Two deployment-wide surfaces are settled in ``config.yml`` rather than in a
compose template: which image each service runs, and how the ``${VAR}``
placeholders in the rendered compose files are filled in. Declaring a service
in the first place, and the env chain those placeholders resolve against, are
covered in :doc:`/how-to/deploy-project/index` and
:doc:`/how-to/deploy-project/env-chain`.

.. _deployment-image-overrides:

Overriding Service Images
~~~~~~~~~~~~~~~~~~~~~~~~~

Every service image resolves through the same three-layer chain — an
environment variable wins, then a ``config.yml`` key, then the packaged
default. Thirteen images, one row each:

.. list-table::
   :header-rows: 1
   :widths: 20 30 28 22

   * - Service
     - Environment variable
     - Config key
     - Packaged default
   * - postgresql
     - ``OSPREY_POSTGRES_IMAGE``
     - ``services.postgresql.image``
     - upstream pin
   * - openobserve
     - ``OSPREY_OPENOBSERVE_IMAGE``
     - ``services.openobserve.image``
     - upstream pin
   * - mongodb
     - ``OSPREY_MONGODB_IMAGE``
     - ``services.mongodb.image``
     - upstream pin
   * - event_dispatcher
     - ``OSPREY_DISPATCH_IMAGE``
     - ``services.event_dispatcher.image``
     - ``<project>-dispatch``
   * - dispatch_worker
     - ``OSPREY_WORKER_IMAGE``
     - ``services.dispatch_worker.image``
     - ``<project>``
   * - nextcloud_bridge
     - ``OSPREY_NEXTCLOUD_BRIDGE_IMAGE``
     - ``services.nextcloud_bridge.image``
     - ``<project>-nextcloud-bridge``
   * - gchat_bridge
     - ``OSPREY_GCHAT_BRIDGE_IMAGE``
     - ``services.gchat_bridge.image``
     - ``<project>-gchat-bridge``
   * - bluesky
     - ``OSPREY_BLUESKY_BRIDGE_IMAGE``
     - ``services.bluesky.image``
     - ``<project>-bluesky-bridge``
   * - bluesky (Tiled sidecar)
     - ``OSPREY_TILED_IMAGE``
     - ``services.bluesky.tiled_image``
     - upstream pin
   * - bluesky (Redis sidecar)
     - ``OSPREY_BLUESKY_REDIS_IMAGE``
     - ``services.bluesky.redis_image``
     - upstream pin
   * - bluesky_web
     - ``OSPREY_BLUESKY_WEB_IMAGE``
     - ``services.bluesky_web.image``
     - ``<project>-bluesky-web``
   * - virtual_accelerator
     - ``OSPREY_VA_IMAGE``
     - ``services.virtual_accelerator.image``
     - ``<project>-va``
   * - qmd
     - ``OSPREY_QMD_IMAGE``
     - ``services.qmd.image``
     - ``<project>-qmd``

Point either of the first two layers at an internal registry mirror or a
pinned digest when your deployment host cannot (or should not) pull public
images.

Five of the thirteen are **upstream pins** — images somebody else publishes,
named exactly as they publish them. The other eight are **built by OSPREY**
from your project, and their default reference is assembled rather than
fixed: a project name, a per-service suffix, and the two axes below.

.. _deployment-image-axes:

The two image axes
^^^^^^^^^^^^^^^^^^

An OSPREY-built default is always spelled the same way::

   <registry>/<project><service suffix>:<tag>

Two stack-wide settings supply the ends of that name, so an entire deployment
can be moved to a registry — or to a different tag — without touching any of
the thirteen rows above:

.. list-table::
   :header-rows: 1
   :widths: 16 30 24 30

   * - Axis
     - Environment variable
     - Config key
     - When neither is set
   * - Registry
     - ``OSPREY_IMAGE_REGISTRY``
     - ``images.registry``
     - no prefix at all
   * - Tag
     - ``OSPREY_IMAGE_TAG``
     - ``images.tag``
     - ``local``

.. code-block:: yaml

   # config.yml — every OSPREY-built image comes from the mirror,
   # at the tag the pipeline pushed
   images:
     registry: registry.example.org/accelerator
     tag: "2026.08.1"

.. code-block:: bash

   # or, for one build
   OSPREY_IMAGE_REGISTRY=registry.example.org/accelerator \
     OSPREY_IMAGE_TAG=2026.08.1 osprey build

For each axis the environment variable wins, then the config key, then the
packaged default. A blank value counts as unset on both layers — an
exported-but-empty variable is how a shell spells "I did not set this", so a
stray ``OSPREY_IMAGE_TAG=`` cannot render an image reference with no tag. A
trailing slash on the registry is optional: one is added if you leave it out,
and never doubled if you put it in.

Two things about *when* and *where* this applies are worth having straight:

* **The axes are the innermost layer, not an override.** They decide what the
  packaged default of an OSPREY-built image is. A ``services.<name>.image``
  pin still beats them for that one service, and an ``OSPREY_<SVC>_IMAGE``
  variable still beats both. Setting an axis moves everything you have not
  pinned individually.
* **The axes are read when the compose files are rendered**, not when the
  containers start. Export them for the ``osprey build`` that produces the
  deployment; the per-image ``OSPREY_<SVC>_IMAGE`` variables, by contrast, are
  filled in by compose at ``osprey up`` time. With neither axis set the render
  is what it always was — ``<project>:local`` and its siblings — so a
  deployment that never heard of them is unaffected.

The axes never touch the five upstream pins. Prefixing ``mongo:7`` with your
registry would name an image that exists in no registry; mirror those through
their own row instead.

.. _compose-interpolation-precedence:

Where ``${VAR}`` values come from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compose fills in ``${VAR}`` placeholders in the rendered compose files from two
sources: the env file(s) the command passes, and the environment of the shell
you typed the command in. **The two supported providers disagree about which of
those wins**, and the disagreement is a straight inversion:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Provider
     - What gets substituted when both sources set a variable
   * - Docker Compose v2
     - The **exported shell value**. ``--env-file`` is its lower-precedence
       source, so a variable exported in your shell overrides the env chain.
   * - podman-compose
     - The **env-file value**. It resolves from the env file before the calling
       shell, so the export reaches nothing.

You do not have to remember this. ``osprey up`` compares the two sources on
every start, and when an exported variable disagrees with the value the env
chain resolves to it warns by name — never by value — and states which value
the provider it just probed will actually use, plus what to do about it. The
reliable habit is to put the value in the env chain and leave your shell out of
it: that is the one gesture that means the same thing on both providers.

.. _config-protected-set:

The protected set
-----------------

A closed list of files and config keys may not be rewritten by the running
agent, whichever surface the write arrives through: the rendered
``config.yml``, ``.claude/settings.json``, ``.mcp.json``, ``CLAUDE.md``, the
safety hooks, rules and skills, the limits and plan-device tables — and, inside
``config.yml``,
the key families that gate writes and approval (``control_system.*``,
``approval.*``, ``hooks.*``, ``claude_code.*``, among others). These artifacts
are owned by the build profile: edit them in the profile and run
``osprey build``. Every refused attempt names the owning channel and is
recorded in ``var/audit/protected-writes.jsonl``.

.. seealso::

   :doc:`profile`
      The build profile these settings are rendered from.

   :doc:`environment-variables`
      Settings that arrive from the environment rather than from ``config.yml``.

   :doc:`/how-to/health-and-monitoring/configure-health-checks`
      Putting the ``health:`` block to work, and the ``SYSTEM`` dashboard.

   :doc:`/how-to/web-terminal/operate`
      Running the terminal the ``web:`` settings render.
