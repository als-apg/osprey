===================================
Production Control Systems Tutorial
===================================

The :doc:`Hello World Tutorial <hello-world-tutorial>` built a single-server agent
that reads channels by their exact PV name. A production control room needs more:
operators think in terms of *devices and physics* — "the booster's defocusing
quadrupole" — not PV strings; they need to search the electronic logbook, pull
historical trends, and produce shift reports. This tutorial builds the
**control-assistant** agent, which adds those capabilities on top of the same
mock control system and the same safety model.

By the end you'll have an agent that finds channels from natural-language
descriptions, searches a seeded operations logbook, plots archived data, and runs
control-room operator skills.

.. dropdown:: **Prerequisites**
   :color: info
   :icon: list-unordered

   **Required:**

   - Python 3.11+
   - `Claude Code <https://docs.anthropic.com/en/docs/claude-code/overview>`_ CLI installed
   - Osprey framework installed (``uv tool install osprey-framework``, or
     ``uv sync --extra dev`` if working from a clone)
   - ``ANTHROPIC_API_KEY`` set in your environment

   **Recommended:**

   - Finish the :doc:`Hello World Tutorial <hello-world-tutorial>` first. This
     tutorial assumes you already understand the mock connector, the safety
     limits system, and the human-approval flow, and it does **not** re-teach
     them.

.. note::

   New to Osprey? Start with the :doc:`Hello World Tutorial <hello-world-tutorial>`.
   It covers project layout, the ``controls`` MCP server, and the safety model
   in detail — everything on this page builds directly on it.

Step 1: Build the Project and Bring It Up
------------------------------------------

Create a project from the ``control-assistant`` preset:

.. code-block:: bash

   osprey init my-control-assistant --preset control-assistant
   cd my-control-assistant
   osprey build

As in Hello World, this writes one repository whose ``profile.yml`` is the
source and whose ``build/`` is the render; provider keys go in the repository's
``.env``.

No hardware is involved, so every example below is safe to run; hardware writes
are still gated behind the human-approval prompt. Unlike ``hello-world``, the
preset ships a **channel database**, a **seeded electronic logbook**, and an
**archive of its own** — a real store this project deploys, seeds and records
into — plus a set of sub-agents and operator skills.

.. note::

   First run may take 1--2 minutes to create a virtual environment and install
   dependencies. Subsequent builds are near-instant.

Then bring the stack up (this needs Docker or Podman):

.. code-block:: bash

   osprey up

This starts the containers the preset declares: the Virtual Accelerator the
agent reads and writes — a containerized simulator with LUME-backed physics, laid
out in :doc:`../architecture/virtual-accelerator` — and the **archive**, a
MongoDB store and a recorder service that holds what those channels did.

**The first deploy seeds the archive**, and it is the step that takes the
longest. It writes about a month of history for every channel the machine
serves, reporting a step under the start phase every 15 seconds or so while it
works. The duration at the end of each step is how long that slice took:

.. Renderer-generated block, not a captured run: a real seed needs a container
   runtime, the store, and several minutes, so the shape below was produced by
   printing the deploy path's own strings through the real phase reporter. The
   counts and durations are historical, carried over from an earlier capture.
   To regenerate: build each line the way ``container_lifecycle`` does (the two
   ``_report_step`` literals around the seed, plus
   ``SeedReport.describe()`` for the last one) and print them through
   ``Phase.step`` on a ``LiveReporter``.

.. code-block:: text

   → Starting my-control-assistant
     · seeding the archive base: 2,908 channels over 30 days (minutes on a first deploy)
     · seeding archive: 8,960 documents written across 2,908 channels (17.4s)
     · seeding archive: 17,920 documents written across 2,908 channels (15.0s)
     ...
     · archive base: seeded 57,600 documents x 2,908 channels (2026-07-12 09:14 to 2026-08-11 09:14 UTC) in 96.3s (14.2s)

Each of those documents holds one instant across every channel, which is why
the count is in the tens of thousands rather than the millions. Expect a minute
or two on a first deploy — it is writing, not hanging. Later deploys check what
is already stored against the settings now in force and skip the seed when it
already matches.

.. note::

   The store is compressed (zstd) and bounded by its retention window: at the
   shipped settings it stays under 2 GiB on disk, and samples expire rather than
   accumulating forever. Tuning that is Step 7.

Step 2: What's Different from Hello World
------------------------------------------

The ``control-assistant`` preset keeps the same ``controls`` MCP server and the
same safety hooks, then adds production capabilities. The most visible additions:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Capability
     - What it adds
   * - **Channel finder** (sub-agent)
     - Resolve channels from a natural-language description by exploring a channel
       database — no need to know PV names.
   * - **Logbook search** (sub-agents)
     - ``logbook-search`` and ``logbook-deep-research`` query a seeded operations
       logbook for past events.
   * - **Archive + visualization**
     - A store this project deploys serves historical data — seeded on the first
       deploy, then recorded from the running machine; the ``data-visualizer``
       sub-agent turns it into interactive and publication-quality plots.
   * - **Operator skills**
     - ``/diagnose``, ``/session-report``, ``demo-gallery``, and ``demo-ui``
       support common control-room workflows.
   * - **Web terminal**
     - A browser split-pane UI with logbook and channel-finder panels
       (documented separately — see :doc:`../how-to/web-terminal/operate`).

The **safety model is unchanged**: limits checking, the pre-write check, and the
human-approval prompt all behave exactly as in the Hello World Tutorial. Refer
back to that tutorial's "Write with the Guards Watching" step for the
write-approval walkthrough — we won't repeat it here.

Start the agent from anywhere inside the project — this tutorial stays in the
terminal with ``osprey chat``; the web terminal from Hello World
(``osprey web``) works just as well:

.. code-block:: bash

   osprey chat

.. note::

   On first run, the Osprey agent will ask you to trust the MCP servers in this
   project. Accept to allow the agent to use the control system and channel-finder
   tools.

Step 3: Find Channels by Description
-------------------------------------

This is the headline capability. In Hello World you had to know the exact PV:

.. code-block:: text

   You: Read channel SR:BEAM:CURRENT

With the control assistant, you describe the channel in plain language and the
agent figures out the PV for you:

.. code-block:: text

   You: Read the current in the booster's defocusing quadrupole

The main agent delegates to the **channel-finder** sub-agent, which explores the
channel database and resolves your description to a real PV before reading it.
The exact wording depends on your ``CLAUDE.md`` and output style, but you'll see
something like:

.. code-block:: text

   Resolved "booster defocusing quadrupole current" → BR:MAG:QD:01:CURRENT:RB

   Channel: BR:MAG:QD:01:CURRENT:RB
   Value:   142.7 A
   Status:  OK

Try a few more descriptions — none of them require you to know the PV naming
scheme:

.. code-block:: text

   You: What's the storage-ring beam current?

.. code-block:: text

   You: Show me the horizontal corrector magnet setpoints in the storage ring

The first resolves to ``SR:DIAG:DCCT:01:CURRENT:RB`` (the storage-ring DC current
transformer); the second resolves to the ``SR:MAG:HCM:*:CURRENT:SP`` family.

**How it works.** The channel database is organized as a hierarchy
(``ring : system : family : device : field : subfield``), and the channel-finder
agent navigates it systematically: it checks prior examples
(``view_examples``), explores the available options at each level
(``get_options``), then constructs and validates the final addresses
(``build_channels``). It **never fabricates PV names** — every channel it returns
comes from the database. For a deeper look at the finder and its other search
strategies, see :doc:`../how-to/use-channel-finder`.

Step 4: Search the Electronic Logbook
--------------------------------------

The control-assistant bundle seeds an electronic logbook with realistic
operations entries — RF trips, vacuum maintenance, beam-recovery shifts,
radiation surveys, and more. Ask the agent about past events in natural language:

.. code-block:: text

   You: Search the logbook for RF cavity temperature trips

The agent delegates to the **logbook-search** sub-agent, which matches the query
against the logbook and summarizes what it finds:

.. code-block:: text

   Found 3 related entries:
   - "Beam dump — RF cavity C1 reflected power trip (thermal excursion)"
   - "Investigation: Recurring cavity C1 thermal excursions"
   - "RF cavity C1 cooling manifold repair"

   Summary: Cavity C1 tripped on reflected power after its temperature climbed;
   a follow-up investigation traced it to reduced cooling-water flow, later fixed
   by a manifold repair.

For questions that span several entries and need synthesis (for example, *"trace
the root cause of the recurring C1 trips and what finally resolved them"*), the
agent can use the **logbook-deep-research** sub-agent, which performs a multi-hop
search and stitches the entries into a single narrative.

Step 5: Pull Historical Data and Plot It
-----------------------------------------

The archive Step 1 seeded is what answers history questions here — the agent
reads stored samples, not numbers made up when you ask. Ask for a trend:

.. code-block:: text

   You: Plot the storage-ring beam current over the last 24 hours

The agent finds the channel (``SR:DIAG:DCCT:01:CURRENT:RB``), reads its history
from the archiver, and delegates to the **data-visualizer** sub-agent to render
the result. The visualizer produces a self-contained figure artifact:

.. code-block:: text

   Read 8640 samples for SR:DIAG:DCCT:01:CURRENT:RB (last 24h).
   Created interactive plot: beam_current_24h.html (artifact)

The ``data-visualizer`` can produce interactive Plotly figures, publication-quality
matplotlib images, dashboards, and LaTeX reports. Here the query hits the store
this project deploys; in production the same query hits your facility's archiver
(see Step 7).

Two things follow from the history being *stored* rather than made up on
demand. The last 24 hours come back at one sample every 10 seconds, because that
is the cadence the archive holds recent history at — so a 24-hour trend is
around 8,600 points, not a smooth line drawn to fit your window. And a question
reaching back further than the archive does gets **no points** rather than a
plausible-looking answer: ask for last year and the honest reply is that the
archive starts about a month ago.

Step 6: Run Operator Skills
----------------------------

The preset installs control-room **skills** you invoke directly. A few worth
trying:

**Generate a shift report** --- summarize the session's actions into a polished,
self-contained HTML report:

.. code-block:: text

   You: /session-report

The skill asks what kind of report you want (chronological log, technical
analysis, or executive briefing), gathers the session's artifacts, and saves an
HTML report to the artifact gallery.

**Triage an infrastructure failure** --- when a tool call or server misbehaves:

.. code-block:: text

   You: /diagnose

``/diagnose`` investigates Osprey infrastructure problems (failed tool calls,
connection errors, configuration drift) and produces a structured root-cause
report — it is for diagnosing the *assistant*, not the accelerator.

**Other skills:** ``demo-gallery`` generates a showcase of plot and report
artifacts to explore the gallery's capabilities, and ``demo-ui`` runs a short
scripted demonstration of the agent driving the web workspace — switching
panel tabs, focusing artifacts, composing layouts.

.. note::

   Editing this deployment's own configuration is deliberately not one of the
   skills here. The preset withholds the ``setup-mode`` skill and the agent's
   ``setup_patch`` tool from the control-room agent — including the one this
   tutorial runs — because rewriting the project is administration rather than
   control-room work. The multi-user stack grants both to a single admin
   login; see :ref:`the tier table <multi-user-tiers>`. To change something in
   this tutorial, edit ``profile.yml`` and run ``osprey build``, which is what
   the rest of this page does.

.. note::

   This project also ships a browser-based **web terminal** with logbook
   and channel-finder panels. It has its own guide —
   see :doc:`../how-to/web-terminal/operate` to launch it with ``osprey web``.

Step 7: Tune the Assistant and Go to Production
------------------------------------------------

**Choose a channel-finder strategy.** The preset defaults to ``hierarchical``
mode, which scales to large facilities with thousands of channels. For a small
facility (under ~1,000 channels) the ``in_context`` strategy can be simpler and
faster. The strategy is a build-time choice, so set it and rebuild:

.. code-block:: bash

   osprey set channel_finder_mode=in_context
   osprey build

``osprey set`` writes the value into ``profile.yml``, so it stays in effect for
every later build — you can also just edit that file instead.

See :doc:`../how-to/use-channel-finder` for a comparison of the strategies.

**Tune how much history the archive keeps.** Two settings decide the store's
size and reach: ``retention_days`` (how far back it goes — 30 by default) and
``hot_span_hours`` (how much of that is kept at the dense 10-second cadence —
48 by default; everything older is kept at 60 seconds). Both live in the
**profile**, in its ``va_archiver:`` block, which is the one place the archive is
described:

.. code-block:: yaml

   # profile.yml
   va_archiver:
     retention_days: 7        # a week is plenty for a demo, and seeds faster
     hot_span_hours: 24

Rebuild and deploy after editing. The deploy notices the stored history no
longer matches what the profile asks for, says what changed, and reseeds it
(pass ``--keep-archiver-base`` to leave the old data alone instead). Do **not**
copy these keys into the project's ``config.yml`` — the build writes that file
from the block, and a second copy is free to disagree with the first.

**Switch to real hardware.** As in Hello World, moving to production is a
configuration change, not a code change. Point the connectors at your facility
in ``profile.yml``:

.. code-block:: bash

   osprey set connector=epics
   osprey set config.archiver.type=epics_archiver
   osprey set va_archiver=null

The archive this tutorial deploys is a *simulated* machine's history, which is
not what you want against hardware — so the archiver moves to your facility's
appliance at the same time as the control system, and the recorded store is
dropped. All three lines are needed: the build refuses a facility baseline that
still carries a ``va_archiver:`` block, because that store would be served as
the real machine's past.

Because these are build-time inputs, re-render the agent's artifacts and
relaunch:

.. code-block:: bash

   osprey build
   osprey chat

Your queries don't change --- "Read the current in the booster's defocusing
quadrupole" and "Plot the storage-ring beam current over the last 24 hours" now
run against live EPICS and your real archiver. The connectors handle the
difference; the agent, the channel finder, and your prompts stay the same.

Next Steps
==========

You've built a production-shaped control assistant with channel finding, logbook
search, historical plotting, and operator skills. Where to go next:

- **Run your first plan**: this project can run real measurement plans — ask
  the agent for a grid scan, review it in the BLUESKY panel, and start it from
  the BLUESKY panel, all against the Virtual Accelerator.
  :doc:`../how-to/bluesky/run-first-plan` walks through it in ten minutes.
- **Channel finder in depth**: :doc:`../how-to/use-channel-finder` compares the
  hierarchical, in-context, and middle-layer strategies and explains the database
  format.
- **Web terminal**: :doc:`../how-to/web-terminal/operate` launches the browser UI and
  its panels with ``osprey web``.
- **Tailor a preset to your facility**: :doc:`../how-to/build-profiles` shows how
  to turn ``control-assistant`` into a profile you own and edit.
- **Architecture deep dive**: the :doc:`conceptual-tutorial` and the
  :doc:`Architecture <../architecture/index>` section explain the agent + MCP
  design, the connector system, and the safety mechanisms.
- **CLI reference**: see :doc:`/reference/cli` for all ``osprey`` commands.
