.. _how-to-apps:

==========================
Build Your Own OSPREY Apps
==========================

An **app** is your deployment's own tool for one task on the machine — solve a
local orbit bump, show and set a channel, step a magnet through a table — used
by the operator from a page in the web terminal and by the agent from its
tools. It is how a facility extends OSPREY with its own workflows without
changing OSPREY itself: the app lives in your deployment repository, is built
and started with the rest of the deployment, and writes to the machine only
through the same chain as every other write.

An app has three parts, all described by one file, ``apps/<name>/app.yml``:

- a **backend container** — a small web service, in Python, that holds the
  app's code and state, reads the machine and writes it;
- a **panel** — the page the operator opens from the sidebar, served by the
  backend and shown inside the web terminal;
- a **tool server** — an MCP server that gives the agent tools for the same
  task, each tool calling the backend.

Panel and tools share one backend, so what the agent computes is what the
operator sees in the panel, and what the operator sets in the panel is what the
agent reads back.

.. raw:: html
   :file: ../../_diagrams/apps.html

``osprey build`` turns ``app.yml`` into the three parts; ``osprey up`` starts
them with the rest of the deployment. The ``demo`` preset ships
one app, :doc:`orbit bump <orbit-bump>`, which this section uses as its
example; ``osprey app new`` copies a minimal scaffold for your own.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What an app looks like to the operator and to the agent
   - When a task is an app, and when it is a skill, a Python run or a
     Bluesky plan instead
   - What you write — one manifest, one Python package, one page — and
     what OSPREY supplies
   - How an app's writes stay inside the login's tier, the write switch and
     the approval card
   - The path from ``osprey app new`` to a running app
     (:doc:`build-your-own-app`) and a real app read end to end
     (:doc:`orbit-bump`)

   **Prerequisites:** a deployment repository (``osprey init``) and the
   terms from :doc:`/how-to/web-terminal/multi-user/tiers`: a **login** opens
   a terminal with a **tier**, and the operator's **write switch** per machine
   decides whether a write is refused, asked or allowed.

What the operator sees
======================

In the web terminal's sidebar, under **Apps**, the app is a row with its
label — *Orbit bump* for the example. Clicking it opens the panel as a tile
beside the chat; the panel takes the terminal's theme, light or dark, because
it uses the terminal's design tokens. The row dims while the backend does not
answer its health check and lights up again when it does
(:doc:`/how-to/web-terminal/panels`).

The panel talks to the backend for that operator: the terminal passes the
login's name along, so the backend answers from the machine that login's
session is on — the one named on the :ref:`control-target chip
<web-terminal-session-posture>` — and shows that login's plan, not another's.

A write started from the panel (orbit bump's *Apply…*) opens the panel's own
confirmation with the exact setpoints. A write the agent starts shows in the
chat as an **approval card** when the operator's switch for that machine is at
*asked*: the card names the app, the tool, the machine (*Live — writes move
hardware*, *Simulator — nothing moves*), the switch position, and then the
text the app wrote for it, filled in from the backend's state — for orbit
bump, the plan id and one line per corrector with its current and new value.
The operator approves or refuses. At *allowed* the same write runs without a
card; at *off* it is refused.

On a login whose tier is ``never``, and on every shared login, the panel is
read-only: it shows state, and the terminal refuses the panel's write requests
before they reach the backend.

What the agent sees
===================

The app's tools, named ``mcp__<app>__<tool>`` — ``mcp__orbit-bump__plan_bump``
and ``mcp__orbit-bump__apply_bump`` for the example — beside OSPREY's own
tools. Each tool's docstring is what the agent reads to decide when to call it,
and the manifest says which tools only read and which write:

- a **read tool** runs like any other tool;
- a **write tool** is listed under ``approval`` in ``app.yml``, so every call
  meets the tier, the write switch and the approval before the tool runs.

An app can also carry a **skill**: a ``SKILL.md`` that tells the agent when to
use the app and how to read its answers. Orbit bump's skill says, for instance,
to compute first, show the operator the plan, and only then ask to apply it.
The build installs the skill into every login that shows the app.

The agent reaches the backend only at the routes the manifest declares, through
the web terminal, under the login's name — never directly, never as another
login. When the backend does not answer or refuses a call, the agent sees the
sentence you wrote for that case in ``app.yml`` (``errors.unreachable``,
``errors.refused``).

When to build an app
====================

An app is the right shape when a task needs **its own state and its own view,
for the operator and the agent alike, the same in every session**. Other
shapes cover the rest:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - You want
     - Build
   * - A one-off calculation, plot or check
     - Nothing. Ask the agent; it writes and runs Python under the same tier
       and switch (:doc:`/architecture/python-executor`).
   * - To tell the agent how to do a task with the tools it already has
     - A skill — instructions, no code of your own runs
       (:doc:`/contributing/agent-skills`).
   * - A measurement sequence with a queue, a run record and live plots
     - A Bluesky plan (:doc:`/how-to/bluesky/index`).
   * - An existing web page of your facility in the sidebar
     - A URL-backed panel (:doc:`/how-to/web-terminal/panels`) — nothing to
       write.
   * - A task with state (a plan, a table, a model), a page to work it from,
       and tools so the agent can do it too
     - An app — this section.

The orbit bump app shows the difference. A bump is not one write: it is a plan
— a model, a set of BPMs, the correctors a rule picks, the predicted orbit, a
limit verdict per setpoint — that the operator inspects and the agent reasons
about before anything moves. The plan has to be the same object for both, has
to survive a page reload, and has to be the thing the approval card binds the
write to. That is an app.

What you write
==============

One directory in your deployment repository. ``osprey app new <name>`` writes
all of it from the scaffold, with your name in place:

.. code-block:: text

   apps/corrector-trim/
     app.yml                 the manifest: name, container, panel, tools
     pyproject.toml          the Python package the backend and tools are in
     corrector_trim/
       __init__.py           the app's name and the routes the two sides share
       server.py             the backend: a FastAPI service, serves the panel,
                             answers state, performs the write
       tools.py              the tool server: one function per tool
     panel/
       index.html            the page, using the terminal's design tokens
       panel.js              reads the backend's state and shows it
       panel.css
     tests/
       panel.test.js         Vitest over the panel

Everything in it is ordinary FastAPI, JavaScript and YAML. What OSPREY
supplies, so the app does not have to:

- **the connection to the machine** — the backend asks for a connector and
  reads or writes channels through it; the connector applies the write
  verdict, the limits and the audit record;
- **who is asking** — the terminal passes the login's name to the backend,
  and the SDK turns it into that login's control target;
- **the panel's place** — the sidebar row, the tile, the theme, the health
  dimming, and the proxy that keeps the panel reachable only through the
  terminal;
- **the tool server's wiring** — ``serve_tools`` runs your functions as an
  MCP server with the right name, and the approval card, the stamp and the
  re-check on the tool side come from the manifest's ``approval`` entry;
- **the build** — the wheel, the image, the container, the port, the config
  file, the state directory, the skill's installation per login.

App code imports OSPREY only through ``osprey.apps.sdk`` (plus
``osprey.facility.kinds`` and the public ``osprey_connectors`` modules). Every
name in the SDK keeps its meaning across releases; ``osprey app check`` reports
any other import, so an app built today still builds on the next OSPREY.

How writes stay safe
====================

An app adds no new way to write to the machine. Whether the write starts in
the panel or in a tool:

1. **The tier** of the login decides whether the app may write at all
   (``control_system.writes: never | asked | allowed``). On ``never``, and on
   a shared login, the terminal refuses the panel's write requests and the
   agent's write tools are denied.
2. **The write switch** the operator set for the session's machine decides
   each write: *off* refuses, *asked* shows the approval card, *allowed* runs
   without one.
3. **The approval** binds the write to what was approved: the app's manifest
   says what the card shows and which values of the backend's state the call
   is held to — orbit bump binds the plan id and a digest of its setpoints, so
   a plan that changed after approval is refused.
4. **The connector** judges the write once more — verdict, limits, read-back
   — and records it, as it does for every write in OSPREY
   (:doc:`/architecture/safety-chain`).

:ref:`Writes and approvals <how-to-apps-writes>` on the build page says how to
declare a write tool and what each key does.

Where to go next
================

- :doc:`build-your-own-app` — from ``osprey app new`` to a running panel and
  tools on your deployment: the manifest, the backend, the tool server, the
  panel, writes and approvals, logins and skills.
- :doc:`orbit-bump` — the bundled example, read as an operator, as the agent,
  and as its source.
- :ref:`osprey app <cli-osprey-app>` — the command reference.

.. toctree::
   :hidden:

   build-your-own-app
   orbit-bump
