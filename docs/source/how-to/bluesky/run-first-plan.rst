===================
Run Your First Plan
===================

In this guide you ask the agent for a plan in plain words, watch the form
fill itself in, press one button, and watch the points land. Everything runs
against the Virtual Accelerator, a simulated machine, so nothing real can move.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Asking for a grid scan the way an operator would say it
   - Reviewing and adjusting the plan in the **PLAN** panel
   - What **Run** does, and when it reads **Add to queue** instead
   - Where the results appear, and how to stop a plan at any moment

   **Prerequisites:** the demo tutorial project
   (:doc:`/getting-started/demo`). It ships the whole plan stack
   ready to go, with the Virtual Accelerator as its default machine. There is
   nothing to configure first.

Step 1: Ask
===========

Open the Web Terminal (``osprey web``). Ask for the plan the way you would
say it out loud, in setpoints and monitors, not in software terms:

.. code-block:: text

   Set up a 2-D grid scan: sweep the horizontal steering correctors in
   sectors 1 and 2 from -0.5 to 0.5 A in 5 steps each, reading the BPMs
   either side of them at every point.

The agent turns this into a ``grid_scan`` plan and stages it as a **shared
draft**. A shared draft is a plan in progress that you and the agent both see
and can both edit. Composing a draft never moves anything and never needs
permission.

Step 2: Watch it fill
=====================

Switch to the **PLAN** tab. If the draft is on the plan you are looking at, a
small button appears: *"Draft is now on grid_scan — click to view"*. Click
it, and the form fills with the agent's draft. From here every field the
agent sets glows briefly as it lands, with a short note naming what changed.

The form is yours too. Change a step count or swap a monitor, and your edit
flows back into the same draft the agent sees. Nobody's version wins by
surprise, because there is only one draft. That also means the **Discard
shared draft** button deletes the draft for everyone, the agent included. Use
it only for a real fresh start.

Step 3: Run it
==============

Click **Run**, then confirm. The panel puts *exactly the plan on your screen*
into the plan queue. If anyone changed the draft in the meantime, the panel
refuses and shows you the current version to review instead.

The tutorial project's queue comes up armed, so the plan starts right away. The
button's tooltip says so:

*Runs now.*

The panel offers an **Open BLUESKY** button that takes you straight to the
run.

If the button reads **Add to queue** instead, the queue is stopped, because
someone pressed Stop or Abort earlier. Your plan waits in line. To run it,
switch to the **BLUESKY** tab and press **Start**. Start runs everything in the
queue, in order, not only your item, so check the list first.

.. note::

   Sending a plan toward the machine is the guarded step, whichever button
   does it. On a deployment that has not been armed for execution, the panel
   refuses the run and explains why in plain language. Composing stays free
   because running is guarded. :doc:`queue` explains what "armed" means here.

Step 4: Follow it
=================

On the **BLUESKY** tab, the status chip in the toolbar reads *running*. The
plan sits at the top of the **Queue** view with its progress.

When the plan finishes, it moves to **History** below, and the chip reads
*ready* again. The queue is still armed, so the next plan you run goes
straight through.

Step 5: Watch the results
=========================

Stay on the **BLUESKY** tab. The lower half follows the selected run, and the
run's **figure** leads it. The figure has one plot per panel, with real axis
labels and units, and fills in as the run goes. A 5 × 5 grid settles fast on
the Virtual Accelerator: all 25 points should land within a few seconds.

The raw numbers are underneath, behind a **Data table** disclosure. The
disclosure names the run's row count and stays closed until you open it. A
real run records thousands of rows, so the table stays out of your way by
default. The table shows a preview: a bounded window, labelled with how much
of the run it is holding back.

To get the whole run, click **Export CSV** on the same row. Your browser's
save dialog opens. The saved file carries every row the run recorded, at full
precision.

What you are watching is ``grid_scan``'s own view of the scan. A two-axis grid
draws one **heatmap per monitor**: the fast axis across, the slow axis up, one
cell per grid position. A cell the run has not reached yet is left empty
rather than filled with a zero. If you sweep a single axis instead, the same
plan draws lines: one panel per monitor, against that axis.

Other shipped plans bring their own views too:

- ``orm`` draws a trace per corrector while the sweep runs, then the fitted
  response matrix and per-device scores.
- ``orbit_bump_sweep`` draws the bump itself: the orbit shift across the BPMs
  at each step of the profile. The shift is displaced where you asked for it
  and back on the reference where the bump closes. The residual against its
  tolerance band is drawn under it. The sweep measures; to set one bump and
  leave it, use the :doc:`orbit bump app </how-to/apps/orbit-bump>`.

A short note above the plots says where the data came from and whether it is
still arriving. While the plan runs, it reads *live data · still filling in*.
A run you come back to later, read from durable storage, shows *stored data*
instead.

A plan does not have to bring a view. For a plan that does not, the panel
draws the **default view**: every numeric column the run recorded, plotted
against the run's own axis. The note line adds a few words about why you are
seeing it. None of that is an error. The default view is real data, and for
many measurements it is exactly the right picture.

Ask the agent *"how does that run look?"* and it reads the same figure you
have on screen, so you are both describing one picture rather than two.

If you need to stop
===================

Two buttons on the BLUESKY tab stop a plan, and they always work. No
permission, token or switch can disable them. They sit in the strip at the
top. They show on the Queue tab always, and on every tab whenever anything
runs or the panel cannot tell. Both leave the queue stopped, so the next plan
waits until **Start**.

- **Stop** is gentle. The running plan finishes, then the queue stops.
- **Abort** is immediate. It takes a second, confirming click, because its
  cost is real: the rest of the plan is discarded, and the data already taken
  is kept.

Abort itself does not drive hardware back to a starting position. Whether the
hardware comes back is up to the plan. ``orm`` and ``orbit_bump_sweep`` put
their correctors back on every exit path, an abort included. A plan that makes
no such promise leaves the machine **wherever the plan left it**.

.. dropdown:: What happened behind the scenes
   :color: info
   :icon: gear

   - The draft you watched lives on the **Bluesky bridge**, a small service in
     your project. The agent edits it with its drafting tools; the Plans view
     is a live view of the same object.
   - **Run** pinned the exact draft revision you saw. A revision can be
     queued only once, so a double-click cannot queue a duplicate.
   - A dedicated **queue server** holds the queue itself. It is a separate
     process with its own copy of the devices, so the queue survives restarts
     of everything around it. The tutorial project brings it up armed
     (``bluesky.queue_autostart``), which is why one click ran the plan.
   - Sending a plan toward the machine is checked against a **launch token**
     the deployment holds. That covers **Run** on an armed queue and **Start**
     on a stopped one.
   - Whether the agent holds a launch token too is the deployment's choice.
     Where the deployment grants it, asking the agent to run a plan runs the
     plan: your approval of that one tool call is the decision. Where it does
     not, the agent's request is refused and the agent tells you so. You then
     run the plan from the panel yourself.
   - Running is also switched off entirely for the agent whenever the
     project's control-system writes are disabled.

.. dropdown:: First-run hiccups
   :color: info
   :icon: question

   **A banner says this deployment is browse-only.**
      Your project is a deployment serving the simulator in process, which
      cannot execute plans. You can compose and validate plans, but the queue
      refuses to hold them. The banner names the exact command that serves the
      simulator from its container
      (``osprey set config.control_system.connector.virtual_accelerator.serving=served``); see
      :doc:`/how-to/control-systems/use-virtual-accelerator`.

   **Run or Start is refused.**
      The refusal in the panel says why, in a sentence. The common causes
      are:

      - The deployment is not armed (no launch token).
      - The queue still holds a plan someone stopped earlier. Remove it first;
        see :doc:`queue`.
      - The queue server is still starting up. Wait a moment and try again.

   **Progress reads "N points so far" instead of a percentage.**
      This is expected. Not every plan can predict its total point count, so
      the panel counts what has arrived rather than invent a percentage.

.. seealso::

   :doc:`queue`
      The full picture: what needs arming, reading a refusal, and what
      happens after an emergency stop.

   :doc:`write-plans`
      When the shipped plans aren't enough — add your own.
