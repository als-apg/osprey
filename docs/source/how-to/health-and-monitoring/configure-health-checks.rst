.. _how-to-configure-health-checks:

Configure Health Checks
=======================

``osprey health`` runs a suite of diagnostics over an OSPREY installation and
prints a categorized report. The built-in checks always run; a ``health:``
block in ``config.yml`` — set from the build profile — lets a facility *add*
its own checks (HTTP endpoints, MCP servers, deployed containers,
control-system channels, model providers) and tune the suite's timing.

This guide shows how to put that surface to work. Every ``health:`` field is
catalogued in :ref:`config-health`; the ``--json`` report and its exit codes
are in :doc:`/reference/contracts/health-json`; the full flag list is
``osprey health --help``.

Rows reach one report from six independent places, and three surfaces read the
result at different tiers:

.. raw:: html
   :file: ../../_diagrams/health-suite-composition.html

Cost classes and ``--full``
---------------------------

Every category is either **poll** — cheap and side-effect-free, run on every
``osprey health`` — or **on_demand** — costly or externally-visible (a live
model-chat completion, a package download), run *only* with ``--full``.
Without ``--full`` an on_demand category is reported as a single ``skip`` row
(which still counts toward exit code 0). ``--category NAME`` scopes which
categories run but never elevates cost class:

.. code-block:: bash

   osprey health                              # poll checks only
   osprey health --full                       # poll + on_demand checks
   osprey health --category providers         # just the providers category
   osprey health --full --category model_chat # run the on_demand model-chat category

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

Reads go through the same connector the agent itself uses (selected by
``control_system.type``), so a green canary also proves the connector
configuration end to end.

Recipe: archive freshness, without declaring a check
-----------------------------------------------------

A project that deploys its own archive — one whose build profile carries a
``va_archiver:`` block (see :doc:`../build-profiles`) — can have the freshness
check written for it. Name one canary channel and nothing else:

.. code-block:: yaml

   # in the build profile, not config.yml
   va_archiver:
     freshness_channel: SR:DIAG:DCCT:01:CURRENT:RB

The build derives a complete ``archiver`` category from that, including the
``max_age_s`` you would otherwise have to choose: **three times the recorder's
own sample cadence**, floored at 60 seconds, so the threshold follows the
recorder with no second number to re-tune. Which channel is representative is
the one thing only you know, so there is no default — a profile that names
none derives no check. Declare the check yourself *or* name a
``freshness_channel``, not both: a profile that does both is refused at build
time, because the two would be one fact in two homes.

.. note::

   On a ``mock`` control system the recorder idles by design, so the check
   reports the archive as **stale** — a ``warning``, never an ``error``, and
   an honest answer: the store is reachable, it is simply not being written.

Checks that need real Python
----------------------------

A check that has to query a facility service or compute a derived state cannot
be written in YAML. For those, ``health.plugins`` takes dotted module paths and
loads each module's checks as a category of its own — writing that module is a
developer task, covered in :doc:`/contributing/extending-osprey`.

The web dashboard (``SYSTEM`` panel)
------------------------------------

In a Web Terminal build that ships panels, the same **poll-class** results are
served as a read-only browser dashboard — the ``SYSTEM`` tab. It never runs
``on_demand`` checks, so a browser can never trigger a costly probe; those
categories render as cards carrying a copyable ``osprey health --full`` hint.
The tab appears only when ``system-health`` is in the build's ``web_panels``
list, and its LED reflects sidecar liveness only — the pass/warn/fail status
lives inside the panel.

Title, host, port, and auto-launch live under ``health.title`` and
``health.web`` (all optional):

.. code-block:: yaml

   health:
     title: "Beamline Health"   # dashboard heading (default "System Health")
     web:
       host: 127.0.0.1          # default 127.0.0.1
       port: 8094               # default 8094
       auto_launch: true        # default true

The sidecar re-reads ``config.yml`` and the project ``.env`` on each refresh,
so most edits are picked up on the next poll without a restart. The exception
is ``control_system``: once a connector is open, that change is surfaced as a
notice row rather than applied live — restart ``osprey web`` to pick it up.

The agent surface
-----------------

The agent reads the same suite through two tools, mirroring the CLI's poll /
``--full`` split: ``health_check`` (poll tier — cheap, read-only,
auto-approved) and ``health_check_full`` (the ``on_demand`` tier — costly, so
it asks for approval first, and always runs fresh).

The poll tool serves its answer from a short-lived cache, and each response
says how fresh it is: ``cached``, ``age_s``, and ``refresh_suppressed`` — the
last one ``true`` only when a stuck run made the tool serve its last good
result, in which case treat the answer as possibly stale and ask again
shortly. Each agent session runs its own health server with its own cache;
nothing is shared across sessions.
