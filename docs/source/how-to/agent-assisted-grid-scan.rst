=================================
Agent-Assisted Grid Scan Tutorial
=================================

How to ask the OSPREY agent to set up an n-dimensional ``grid_scan``, watch
the **PLAN** panel fill in live as it works, review and adjust the result by
hand, then add it to the queue and watch points land in **BLUESKY** — all on
the Virtual Accelerator.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Asking the agent for a 2-D corrector grid scan in operator language
   - How the PLAN panel binds to the agent's shared draft and flashes the
     fields it fills
   - Adjusting a field by hand mid-draft — and why that's safe with no arming
   - Why running is two steps — add to the queue, then start it — and where
     points show up

   **Prerequisites:** the Control Assistant tutorial project (see
   :doc:`/getting-started/control-assistant`) with the bluesky scan stack
   enabled. The Virtual Accelerator ships as part of that stack and is the
   tutorial's default connector — see :doc:`use-virtual-accelerator`.

Overview
========

The agent composes a Bluesky scan plan through three drafting tools
(``get_draft`` / ``set_draft`` / ``clear_draft``) against a
single **shared draft** held on the Bluesky bridge. The human's PLAN panel
binds to that same draft: every field the agent sets is broadcast over SSE
and glows in the panel as it lands, and any field the human edits by hand
flows back into the same draft the agent sees. Pressing **Add to queue**
queues exactly the draft revision the panel last showed — nothing the agent
or the human can't see — and a second, deliberate **Start queue** is what
begins real motion.

Prerequisites
=============

Nothing to configure. The Control Assistant preset ships the whole scan stack
enabled — the bluesky MCP server, the PLAN and BLUESKY panels, the queue
server, and the Virtual Accelerator soft-IOC — and ``virtual_accelerator`` is
its default connector, so correctors move and BPMs respond out of the box.
Build and deploy the tutorial project as usual and you are ready.

.. note::

   If your project has been pointed at the ``mock`` connector, scans are
   **browse-only** there: plans compose and validate, but the queue refuses to
   hold them, and the panels say so in a banner naming the command that flips
   back. See :doc:`use-virtual-accelerator`.

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
   plan, so what you are looking at always matches what an Add would queue.

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

Add it to the queue
===================

Once the plan validates, click **Add to queue**, then **Confirm add**. This
queues the *exact draft revision the panel just showed you*: the panel
flushes any pending edit and pins that revision to the request, and the
bridge refuses the request if the draft has moved on since (someone else
edited it, or it was cleared). You get a clear "the draft changed since you
last saw it — refreshed, review and add again" message instead of a
mismatched item. The same pin also makes a double-click harmless — a revision
that has already been queued is rejected rather than queued twice.

Adding does not run anything. The item sits in the queue until someone starts
it.

Start the queue
===============

Switch to the **BLUESKY** tab. Your item is listed under **Queue**; click
**Start queue** to begin. That is the moment hardware moves, and it runs the
queue *as it stands* — every pending item in order, not only the one you just
added. Read the list before you click.

.. note::

   Composing costs nothing and needs no arming; starting does. **Start queue**
   is checked against the **launch token**, and on a stack that is not armed
   the bridge refuses the start and the panel shows its explanation in a
   banner rather than starting anything. The agent's equivalent
   (``queue_start``) is denied outright when
   ``control_system.writes_enabled`` is false. Neither path can start an
   unarmed stack; only the agent path also honours the writes switch. Adding
   to an *idle* queue from the panel needs neither — see
   :doc:`run-scan-queue`.

Results
=======

Stay on the **BLUESKY** tab: the lower half is the selected run's results.
Points appear as the scan runs, one per grid position, with a table and a
live chart of each detector's readings against row order. A 5×5 grid over two
correctors settles quickly on the Virtual Accelerator — you should see all 25
points land within a few seconds of starting the queue.

The same tab carries the two halts: **Stop after current item** lets the
running scan finish and stops the queue there, and **Abort running plan**
stops the scan that is moving hardware right now. Both are always available,
even on a stack with writes disabled.

.. seealso::

   :doc:`run-scan-queue`
      The queue in full: arming, session plans, refusals, stopping.

   :doc:`use-virtual-accelerator`
      Configuring the Virtual Accelerator the tutorial runs against.

   :doc:`/getting-started/control-assistant`
      The tutorial project this guide runs against.
