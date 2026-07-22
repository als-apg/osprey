Configure Health Checks
=======================

``osprey health`` runs a suite of diagnostics over an OSPREY installation —
configuration validity, file-system layout, Python environment, container
infrastructure, telemetry store, API providers, the agent CLI, and any
configured framework services (ARIEL, channel finder) — and prints a
categorized report. The built-in checks always run; the ``health:`` block in
``config.yml`` lets a facility *add* its own checks (HTTP endpoints, MCP servers,
deployed containers, control-system channels, model providers) and tune the
suite's timing.

This guide covers the ``health:`` configuration surface. For the command's flags
and exit codes, see ``osprey health --help``.

Cost classes and ``--full``
---------------------------

Every category is either **poll** or **on_demand**:

- **poll** — cheap and side-effect-free (a socket connect, a version string, a
  single channel read). Poll categories run on every ``osprey health``.
- **on_demand** — costly or externally-visible (a live model-chat completion, a
  package download). On_demand categories run *only* when you pass ``--full``.

Without ``--full``, each on_demand category is reported as a single ``skip`` row
carrying a "run with ``--full``" hint rather than being executed. Selecting a
category with ``--category NAME`` scopes *which* categories run but never
elevates cost class — an on_demand category still needs ``--full`` to actually
execute:

.. code-block:: bash

   osprey health                              # poll checks only
   osprey health --full                       # poll + on_demand checks
   osprey health --category providers         # just the providers category
   osprey health --full --category model_chat # run the on_demand model-chat category

A ``skip`` row does not fail the suite (it counts toward exit code 0), so a
default run stays green even though its on_demand categories were never executed.

The ``health:`` block
---------------------

All configuration lives under a top-level ``health:`` key. Every field is
optional; an absent ``health:`` block runs the built-in checks with their
default timing.

.. code-block:: yaml

   health:
     suite_timeout_s: 30          # poll-class wall-clock budget (default 30)
     on_demand_timeout_s: 120     # on_demand wall-clock budget (default: sum of budgets)
     interval_s: 300              # minimum server-side re-run interval

     plugins:
       - my_facility.health       # dotted module paths (see "Health plugins")

     categories:
       beamline_services:         # a facility-defined category of probe checks
         checks:
           - name: archiver
             type: http
             url: http://archiver.example.com/healthz
           - name: scan_server
             type: mcp
             url: http://localhost:8931/mcp

       providers:                 # metadata-only override of a built-in category
         timeout_s: 15

Probe checks
------------

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
       — this probe checks the data.

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

Recipe: a control-system smoke test
------------------------------------

The single most useful facility check is a canary read of a channel that is
always live on a healthy machine — a beam-current or RF-frequency readback.
Declare it once and it appears as its own category in the CLI report and as a
tile on the web dashboard, graded against the bands you choose:

.. code-block:: yaml

   health:
     categories:
       control_system:
         checks:
           - name: beam_current
             type: channel_read
             address: SR:DCCT
             ok_range: [1.0, 500.0]     # mA — below 1 mA warns (no stored beam)
           - name: rf_frequency
             type: channel_read
             address: SR:RF:FREQ
           - name: archiver_data
             type: archiver_freshness
             channel: SR:DCCT
             max_age_s: 300             # the archiver must have a sample < 5 min old

Reads go through the suite's control-system connector — the same connector,
selected by ``control_system.type``, the agent itself uses — so a green canary
also proves the connector configuration end to end.

Built-in service categories
---------------------------

Beyond the always-on framework checks, two built-in categories are
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

Both are ordinary categories: valid under ``--category``, tunable via a
metadata-only override, and rendered as dashboard tiles with no extra
configuration.

Auto-derived MCP server checks (``health.auto``)
------------------------------------------------

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
  inside a container — a Docker ``/.dockerenv`` marker file, or the
  ``OSPREY_IN_CONTAINER`` environment variable set by the deployment — and uses
  ``docker_url`` when containerized, ``host_url`` on a plain host.

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
--------

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
----------------------------

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
---------------------------

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

Health plugins
--------------

For checks that need real Python — querying a facility service, computing a
derived state — register a **plugin** under ``health.plugins`` as a dotted
module path. The module must expose:

.. code-block:: python

   def get_health_categories() -> dict[str, Callable[[], list[CheckResult]]]:
       """Map category name -> a no-argument callable returning check results."""
       ...

Each callable takes no arguments and returns a list of
``osprey.health.models.CheckResult``; it may be sync or async. Plugin categories
run alongside the built-in and declarative categories through the same path, and
default to ``cost: poll`` (adjust with a metadata override, as above).

Plugin loading is fail-safe: a plugin that fails to import, is missing
``get_health_categories()``, returns the wrong type, or whose category name
collides with a built-in, a declarative, or an earlier plugin category, produces
a single ``error`` row in a diagnostic ``plugins`` category — it never crashes
the suite.

Suite timing
------------

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

The web dashboard (``SYSTEM`` panel)
------------------------------------

In a Web Terminal build that ships panels, the health suite is also served as a
read-only browser dashboard — the ``SYSTEM`` tab. A lightweight sidecar renders
the same **poll-class** results the CLI produces; it never runs ``on_demand``
checks, so a browser can never trigger a costly or externally-visible probe.

Hosting keys
~~~~~~~~~~~~~

The dashboard's title, host, port, and auto-launch live under ``health.title``
and ``health.web``:

.. code-block:: yaml

   health:
     title: "Beamline Health"   # dashboard heading (default "System Health")
     web:
       host: 127.0.0.1          # default 127.0.0.1
       port: 8094               # default 8094
       auto_launch: true        # default true

All are optional; an absent ``health.web`` block serves the dashboard on
``127.0.0.1:8094``.

Enabling the ``SYSTEM`` tab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tab appears only when ``system-health`` is in the build's ``web_panels``
list — panel-shipping presets (for example ``control-assistant``) include it; a
panel-free preset (``hello-world``) does not. Setting ``auto_launch: false``
keeps the tab but does not start the sidecar behind it, so the tab reads as down.

.. note::

   The tab's LED reflects **sidecar liveness only** — green when the health
   sidecar is reachable, red when it is stopped. It is *not* an aggregate status
   light: a check going ``error`` does not turn the LED red. The pass/warn/fail
   status of the suite lives inside the panel, on the ring and the per-category
   cards.

Dashboard behavior
~~~~~~~~~~~~~~~~~~~~

The dashboard polls the sidecar on a cadence derived from ``interval_s``, with a
countdown and a manual refresh. On first open it shows a brief "first scan in
progress" state rather than an error; once the data is behind schedule (older
than ``interval_s``) it surfaces a staleness indicator. ``on_demand`` categories
render as informational cards carrying a copyable ``osprey health --full
--category <name>`` hint — the dashboard has no run buttons, because it never
executes ``on_demand`` work.

Config and ``.env`` edits
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sidecar re-reads ``config.yml`` and the project ``.env`` (the one beside
``config.yml``, exactly as the CLI resolves it) on each refresh, so an edit to
``health.title``, a category timeout, or an ``.env`` value referenced via
``${VAR}`` is picked up on the next poll without a restart — a changed ``.env``
value overrides the previous one, matching CLI semantics.

.. warning::

   Changing ``control_system`` is the exception. Once the sidecar has opened a
   control-system connector (the first time a ``channel_read`` check runs), a
   later ``control_system`` change is **not** applied live: the dashboard shows a
   notice row ("control_system config changed; restart the web terminal to
   apply") and keeps using the original connector. Restart ``osprey web`` to pick
   up the new control system — swapping the connector inside a running process is
   unsafe, so the change is surfaced rather than done silently.

The agent surface
-----------------

The OSPREY agent can read the same health suite through two tools, so you can
ask it "is the facility healthy?" in plain language and it checks for you. The
two tools mirror the CLI's poll / ``--full`` split:

- ``health_check`` — the poll tier. Cheap, read-only, and **auto-approved**: the
  agent runs it without interrupting you. This is the everyday "how are things?"
  check.
- ``health_check_full`` — the full tier, covering the ``on_demand`` checks the
  poll tier reports as skipped (a live model chat, a package-download
  verification). Because those are costly, this tool **asks for your approval
  first**, like any other guarded action.

Reading the freshness fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The poll tool serves its answer from a short-lived cache, and every response
carries three fields that say how fresh it is:

- ``cached`` — ``false`` when the agent just ran the suite, ``true`` when it
  served a recent result without re-running.
- ``age_s`` — how many seconds old that result is (``0`` on a fresh run).
- ``refresh_suppressed`` — normally ``false``. ``true`` is the rare signal that a
  previous run got stuck and the tool served the last good result rather than
  start another; treat that answer as possibly stale and ask again shortly.

A ``health_check_full`` response always reports ``cached: false``, ``age_s: 0``,
and ``refresh_suppressed: false`` — the full tier is always run fresh.

How long a check takes
~~~~~~~~~~~~~~~~~~~~~~~~

The first poll check — and the first one after you edit ``config.yml`` or
``.env`` — runs the whole poll suite inline, so it can take up to
``suite_timeout_s`` (default 30 seconds). After that, checks inside the refresh
window return right away from the cache.

A full check runs everything fresh and takes about as long as the ``on_demand``
checks it covers add up to — bounded by ``on_demand_timeout_s`` (see
`Suite timing`_) plus the poll tier's own budget. Expect it to be the slower of
the two, which is the other reason it is approval-gated.

One server per session
~~~~~~~~~~~~~~~~~~~~~~~~

Each agent session runs its own health server with its own cache. If several
operators are working at once, each session polls independently: three sessions
means three separate poll suites, each still bounded by its own refresh interval
(``interval_s``) and, for the full tier, its own approval prompt. There is no
shared health cache across sessions.
