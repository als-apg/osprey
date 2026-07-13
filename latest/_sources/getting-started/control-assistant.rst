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
   in detail — everything below builds directly on it.

Step 1: Build the Project
--------------------------

Create a project from the ``control-assistant`` preset:

.. code-block:: bash

   osprey build my-control-assistant --preset control-assistant
   cd my-control-assistant

Like ``hello-world``, this project starts in **mock** mode, so every example
below is safe to run with no hardware attached; hardware writes are still gated
behind the human-approval prompt. Unlike
``hello-world``, the preset ships a **channel database**, a **seeded electronic
logbook**, and a **mock archiver**, plus a set of sub-agents and operator skills.

.. note::

   First run may take 1--2 minutes to create a virtual environment and install
   dependencies. Subsequent builds are near-instant.

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
   * - **Archiver + visualization**
     - A mock archiver serves historical data; the ``data-visualizer`` sub-agent
       turns it into interactive and publication-quality plots.
   * - **Operator skills**
     - ``/diagnose``, ``/session-report``, ``setup-mode``, and ``demo-gallery``
       support common control-room workflows.
   * - **Web terminal**
     - A browser split-pane UI with logbook and channel-finder panels
       (documented separately — see :doc:`../how-to/web-terminal/operate`).

The **safety model is unchanged**: limits checking, the pre-write check, and the
human-approval prompt all behave exactly as in the Hello World Tutorial. Refer
back to that tutorial's "Safety & Limits" step for the write-approval walkthrough
— we won't repeat it here.

Start the agent the same way as before, from the project directory:

.. code-block:: bash

   claude

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

The bundle also configures a **mock archiver** that serves synthetic historical
data, so you can exercise the trend-and-plot workflow without a real archive
appliance. Ask for a trend:

.. code-block:: text

   You: Plot the storage-ring beam current over the last 24 hours

The agent finds the channel (``SR:DIAG:DCCT:01:CURRENT:RB``), reads its history
from the archiver, and delegates to the **data-visualizer** sub-agent to render
the result. The visualizer produces a self-contained figure artifact:

.. code-block:: text

   Read 1440 samples for SR:DIAG:DCCT:01:CURRENT:RB (last 24h).
   Created interactive plot: beam_current_24h.html (artifact)

The ``data-visualizer`` can produce interactive Plotly figures, publication-quality
matplotlib images, dashboards, and LaTeX reports. In development everything runs
against the mock archiver; in production the same query hits your real archiver
(see Step 7).

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
artifacts to explore the gallery's capabilities, and ``setup-mode`` inspects and
repairs the project's configuration when something looks misconfigured.

.. note::

   This project also ships a browser-based **web terminal** with logbook
   and channel-finder panels. It has its own guide —
   see :doc:`../how-to/web-terminal/operate` to launch it with ``osprey web``.

Step 7: Tune the Assistant and Go to Production
------------------------------------------------

**Choose a channel-finder strategy.** The preset defaults to ``hierarchical``
mode, which scales to large facilities with thousands of channels. For a small
facility (under ~1,000 channels) the ``in_context`` strategy can be simpler and
faster. The strategy is a build-time choice, so select it when you build:

.. code-block:: bash

   osprey build my-control-assistant --preset control-assistant \
       --set channel_finder_mode=in_context

See :doc:`../how-to/use-channel-finder` for a comparison of the strategies.

**Switch to real hardware.** As in Hello World, moving to production is a
configuration change, not a code change. Point the connectors at your facility in
``config.yml``:

.. code-block:: yaml

   control_system:
     type: epics            # was: mock

   archiver:
     type: epics_archiver   # was: mock_archiver

Because these are build-time inputs, regenerate the agent's artifacts and relaunch:

.. code-block:: bash

   osprey claude regen
   claude

Your queries don't change --- "Read the current in the booster's defocusing
quadrupole" and "Plot the storage-ring beam current over the last 24 hours" now
run against live EPICS and your real archiver. The connectors handle the
difference; the agent, the channel finder, and your prompts stay the same.

Next Steps
==========

You've built a production-shaped control assistant with channel finding, logbook
search, historical plotting, and operator skills. Where to go next:

- **Channel finder in depth**: :doc:`../how-to/use-channel-finder` compares the
  hierarchical, in-context, and middle-layer strategies and explains the database
  format.
- **Web terminal**: :doc:`../how-to/web-terminal/operate` launches the browser UI and
  its panels with ``osprey web``.
- **Tailor a preset to your facility**: :doc:`../how-to/build-profiles` shows how
  to extend ``control-assistant`` with your own overrides and overlays.
- **Architecture deep dive**: the :doc:`conceptual-tutorial` and the
  :doc:`Architecture <../architecture/index>` section explain the agent + MCP
  design, the connector system, and the safety mechanisms.
- **CLI reference**: see :doc:`../cli-reference/index` for all ``osprey`` commands.
