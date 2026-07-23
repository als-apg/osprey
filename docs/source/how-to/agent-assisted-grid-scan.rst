=================================
Agent-Assisted Grid Scan Tutorial
=================================

How to ask the OSPREY agent to set up an n-dimensional ``grid_scan``, watch
the **PLAN** panel fill in live as it works, review and adjust the result by
hand, then Launch and watch points land in **SCAN RESULTS** — all on the
Virtual Accelerator.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Flipping one config line so the demo's scan can actually run
   - Asking the agent for a 2-D corrector grid scan in operator language
   - How the PLAN panel binds to the agent's shared draft and flashes the
     fields it fills
   - Adjusting a field by hand mid-draft — and why that's safe with no arming
   - What launching actually runs, and where points show up

   **Prerequisites:** the Control Assistant tutorial project (see
   :doc:`/getting-started/control-assistant`) with the bluesky scan stack
   enabled, and the Virtual Accelerator container (see
   :doc:`use-virtual-accelerator`).

Overview
========

The agent composes a Bluesky scan plan through three drafting tools
(``get_draft`` / ``set_draft`` / ``clear_draft``) against a
single **shared draft** held on the Bluesky bridge. The human's PLAN panel
binds to that same draft: every field the agent sets is broadcast over SSE
and glows in the panel as it lands, and any field the human edits by hand
flows back into the same draft the agent sees. Pressing **Launch plan**
launches exactly the draft revision the panel last showed — nothing the
agent or the human can't see.

Prerequisites
=============

The mock connector (the tutorial's default) can't run a scan — it doesn't
settle-wait a corrector's readback against its setpoint, which every scan
plan needs between grid points. Point the tutorial at the Virtual
Accelerator instead, which does:

.. code-block:: bash

   osprey config set-control-system virtual_accelerator
   osprey deploy up

That's the only configuration change — the second command re-stages it so the
already-running services pick it up. The soft-IOC itself ships as part of the
stack, so there is no separate container to start; see
:doc:`use-virtual-accelerator` for the details. Everything else (the bluesky
MCP server, the PLAN and SCAN RESULTS panels) ships enabled in the Control
Assistant preset.

Ask the agent
==============

Open the Web Terminal and ask for the scan the way an operator would — in
setpoint and orbit terms, not plan-parameter terms:

.. code-block:: text

   Set up a 2-D grid scan: sweep corrector_01 from -0.5 to 0.5 A in 5 steps
   and corrector_02 from -0.3 to 0.3 A in 5 steps, reading bpm_01 and bpm_02
   at every point.

The agent resolves this into a ``grid_scan`` plan draft — two
``axes`` entries (one per corrector, each naming its ``setpoint`` device as
the bridge knows it, plus ``start``/``stop``/``num_points``) and a
``detectors`` list naming the two BPM readbacks — and
stages the whole thing in a single ``set_draft`` call, noting the ``revision``
it returns. This is staging only: composing the draft never touches hardware,
never requires arming, and never triggers an approval prompt.

Watch it fill
=============

Switch to the **PLAN** tab. If the panel is already showing ``grid_scan``, a
small affordance appears: **"Draft is now on grid_scan — click to view"** —
click it to bind the panel to the draft, which seeds the form from the
current draft state and starts live updates. If the panel is showing a
*different* plan there is no hint; select ``grid_scan`` from the sidebar and
the panel binds to the waiting draft automatically.

.. note::

   An *unbound* panel never silently jumps to a plan you weren't already
   looking at — binding always takes your click or selection. Once bound, the
   panel does follow the draft if the agent switches it to a different
   plan, so what you are looking at always matches what a Launch would send.

Once bound, a **Draft bound** indicator appears next to a **Discard**
button, and every field the agent set (or sets from here) glows briefly as
it lands, with a transient **"agent edited: …"** note naming the changed
keys. Re-sending an already-current value is a silent no-op — no glow, no
note — so only genuine changes draw your eye.

You can adjust any field yourself: change an axis's step count, swap a
detector, anything the form allows. Your edit is sent back into the same
shared draft (a small delta patch, not a full replace), so the agent's next
``get_draft`` sees exactly what you changed. Draft editing — by either
side — never requires arming.

Launch
======

Once the plan validates, click **Launch plan**, then **Confirm launch**.
This launches the *exact draft revision the panel just showed you*: the
panel flushes any pending edit and pins that revision to the request, and
the bridge refuses the launch if the draft has moved on since (someone else
edited it, or it was cleared). You get a clear "the draft changed since you
last saw it — refreshed, review and launch again" message instead of a
mismatched launch. The same pin also makes a double-click harmless — a
revision that has already been launched is rejected rather than run twice.

.. note::

   Draft editing needs no arming, but launching still does. The human and
   agent launch paths are gated differently, and it is worth knowing which
   you are on: this click is checked against the **launch token** only, and
   when that token is unset the panel reports an inert *"writes not armed"*
   rather than launching. The agent's own ``launch_run`` additionally
   requires ``control_system.writes_enabled``. Neither path can launch an
   unarmed stack; only the agent path also honours the writes switch.

Results
=======

Switch to the **SCAN RESULTS** tab. Points appear as the scan runs, one per
grid position, with a table and a live chart of each detector's readings
against row order. A 5×5 grid over two correctors settles quickly on the
Virtual Accelerator — you should see all 25 points land within a few
seconds of confirming the launch.

.. seealso::

   :doc:`use-virtual-accelerator`
      Starting and configuring the Virtual Accelerator container.

   :doc:`/getting-started/control-assistant`
      The tutorial project this guide runs against.
