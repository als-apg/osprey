.. _how-to-configure-health-checks:

Configure Health Checks
=======================

``osprey health`` runs a suite of diagnostics over an OSPREY installation —
configuration validity, file-system layout, Python environment, container
infrastructure, telemetry store, API providers, the agent CLI, and any
configured framework services (ARIEL, channel finder) — and prints a
categorized report. The built-in checks always run; a ``health:`` block in the
build profile (``profile.yml``) lets a facility *add* its own checks (HTTP
endpoints, MCP servers, deployed containers, control-system channels, model
providers) and tune the suite's timing — ``osprey build`` renders it into
``config.yml``.

This guide shows how to put that surface to work. Every ``health:`` field —
the suite scalars, the declarative categories, the six probe types and their
parameters — is catalogued in :ref:`config-health`. For the shape of the
``--json`` report and the exit codes that go with it, see
:doc:`/reference/contracts/health-json`; for the full flag list, see
``osprey health --help``.

Rows reach one report from six independent places, and three surfaces read the
result at different tiers:

.. raw:: html
   :file: ../../_diagrams/health-suite-composition.html

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

Recipe: archive freshness, without declaring a check
-----------------------------------------------------

A project that deploys its own archive — one whose build profile carries a
``va_archiver:`` block (see :doc:`../build-profiles`) — can have the freshness
check written for it. Name one canary channel and nothing else:

.. code-block:: yaml

   # in the build profile, not config.yml
   va_archiver:
     freshness_channel: SR:DIAG:DCCT:01:CURRENT:RB

The build turns that into a complete ``archiver`` category holding one
``archiver_freshness`` check on that channel — the same thing you would write by
hand, plus a ``max_age_s`` you do not have to choose.

**Where the threshold comes from.** It is **three times the recorder's own
sample cadence**, with a floor of 60 seconds. The recorder is what writes this
archive, so how long a healthy archive can go without a sample is a fact about
*it*, not a preference: three intervals tolerates two missed ticks before the
check says anything, which is where "the recorder is behind" stops being
ordinary scheduling jitter, and the floor keeps a fast recorder from alarming
sooner than a container can restart. Slow the recorder down and the threshold
follows — there is no second number to remember to re-tune.

Which channel is representative is the one thing only you know, so there is no
default: a profile that names none derives no check rather than having the
framework guess. The ``control-assistant`` preset names the stored-beam DCCT
current, for the reason a control room would pick it — it is the first thing to
stop moving when the machine does.

.. note::

   On a deployment whose control system is ``mock``, the recorder idles by
   design (it records a virtual accelerator, nothing else), so the newest sample
   is whatever the deploy seeded and the check reports the archive as **stale**.
   That is a ``warning``, never an ``error``, and it is an honest answer rather
   than a misconfiguration: the store is reachable, it is simply not being
   written. Flip the control system back and it goes green within a poll.

Declare the check yourself *or* name a ``freshness_channel`` — not both. A
profile that sets ``freshness_channel`` and also declares
``health.categories.archiver`` in its ``config:`` block is refused at build time,
because the two would be one fact in two homes, merged in whichever order the
keys happened to be read.

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
default to ``cost: poll`` (adjust with a metadata override — see
:ref:`config-health`).

Plugin loading is fail-safe: a plugin that fails to import, is missing
``get_health_categories()``, returns the wrong type, or whose category name
collides with a built-in, a declarative, or an earlier plugin category, produces
a single ``error`` row in a diagnostic ``plugins`` category — it never crashes
the suite.

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
:ref:`config-health`) plus the poll tier's own budget. Expect it to be the slower of
the two, which is the other reason it is approval-gated.

One server per session
~~~~~~~~~~~~~~~~~~~~~~~~

Each agent session runs its own health server with its own cache. If several
operators are working at once, each session polls independently: three sessions
means three separate poll suites, each still bounded by its own refresh interval
(``interval_s``) and, for the full tier, its own approval prompt. There is no
shared health cache across sessions.
