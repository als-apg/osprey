===================
Run Your First Scan
===================

Ask the agent for a scan in plain words, watch the form fill itself in, press
two buttons, and watch the points land — all against the Virtual Accelerator,
a simulated machine, so nothing real can move.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Asking for a grid scan the way an operator would say it
   - Reviewing and adjusting the plan in the **PLAN** panel
   - Why running takes two clicks — **Add to queue**, then **Start queue**
   - Where the results appear, and how to stop a scan at any moment

   **Prerequisites:** the Control Assistant tutorial project
   (:doc:`/getting-started/control-assistant`). It ships the whole scan stack
   ready to go, with the Virtual Accelerator as its default machine — there is
   nothing to configure first.

Step 1: Ask
===========

Open the Web Terminal (``osprey web``) and ask for the scan the way you would
say it out loud — in setpoints and monitors, not in software terms:

.. code-block:: text

   Set up a 2-D grid scan: sweep the horizontal steering correctors in
   sectors 1 and 2 from -0.5 to 0.5 A in 5 steps each, reading the BPMs
   either side of them at every point.

The agent turns this into a ``grid_scan`` plan and stages it as a **shared
draft** — a plan-in-progress that you and the agent both see and can both
edit. Composing a draft never moves anything and never needs permission.

Step 2: Watch it fill
=====================

Switch to the **PLAN** tab. If the draft is on the plan you are looking at, a
small button appears — *"Draft is now on grid_scan — click to view"*. Click
it, and the form fills with the agent's draft. From here every field the
agent sets glows briefly as it lands, with a short note naming what changed.

The form is yours too: change a step count, swap a monitor — your edit flows
back into the same draft the agent sees. Nobody's version wins by surprise,
because there is only one draft. That also means the **Discard shared draft**
button deletes it for everyone — the agent included — so save it for a real
fresh start.

Step 3: Add it to the queue
===========================

Click **Add to queue**, then confirm. This puts *exactly the plan on your
screen* into the scan queue — if anyone changed the draft in the meantime,
the panel refuses and shows you the current version to review instead.

Nothing is running yet. The scan just got in line — and the panel offers an
**Open BLUESKY** button that takes you straight to it.

Step 4: Start the queue
=======================

Click that button, or switch to the **BLUESKY** tab. Your scan is listed under **Queue**. Click
**Start queue** — *this* is the moment things move, and it runs everything in
the queue, in order, not only your item. Glance at the list before you click.

.. note::

   Starting is the guarded step. On a deployment that has not been armed for
   execution, the start is refused with a plain-language explanation in the
   panel — composing and queueing stay free precisely because starting is
   not. :doc:`queue` explains what "armed" means here.

Step 5: Watch the results
=========================

Stay on the **BLUESKY** tab. The lower half follows the selected run: a table
gains one row per grid position, and a live chart traces each monitor's
readings. A 5 × 5 grid settles fast on the Virtual Accelerator — all 25
points should land within a few seconds.

If you need to stop
===================

Two buttons on the BLUESKY tab, and they always work — no permission, no
token, no switch can disable them:

- **Stop after current item** — gentle. The running scan finishes, then the
  queue stops.
- **Abort running plan** — immediate. It takes a second, confirming click,
  because its cost is real: the rest of the scan is discarded, the data
  already taken is kept, and the hardware **stays wherever the scan left
  it** — nothing is driven back to a starting position.

.. dropdown:: What happened behind the scenes
   :color: info
   :icon: gear

   - The draft you watched lives on the **Bluesky bridge**, a small service in
     your project. The agent edits it with its drafting tools; the PLAN panel
     is a live view of the same object.
   - **Add to queue** pinned the exact draft revision you saw. A revision can
     be queued only once, so a double-click cannot queue a duplicate.
   - The queue itself is held by a dedicated **queue server** — a separate
     process with its own copy of the devices. That is why the queue survives
     restarts of everything around it.
   - **Start queue** is checked against a **launch token** the deployment
     holds. For the agent, starting is additionally switched off entirely
     whenever the project's control-system writes are disabled.

.. dropdown:: First-run hiccups
   :color: info
   :icon: question

   **A banner says this deployment is browse-only.**
      Your project is pointed at the ``mock`` control system, which cannot
      execute scans — plans can be composed and validated, but the queue
      refuses to hold them. The banner names the exact command that switches
      to the Virtual Accelerator; see
      :doc:`/how-to/use-virtual-accelerator`.

   **Start queue is refused.**
      The refusal in the panel says why, in a sentence. The common causes: the
      deployment is not armed (no launch token), the queue still holds a scan
      someone stopped earlier (remove it first — see :doc:`queue`), or the
      queue server is still starting up (wait a moment and try again).

   **The chart shows "N points so far" instead of a percentage.**
      That is honesty, not a glitch: not every plan can predict its total
      point count, so the panel counts what has arrived rather than invent a
      percentage.

.. seealso::

   :doc:`queue`
      The full picture: what needs arming, reading a refusal, and what
      happens after an emergency stop.

   :doc:`write-plans`
      When the two shipped plans aren't enough — add your own.
