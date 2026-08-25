===================
Plans and the Queue
===================

Three sentences carry everything on this page. Plans live in a **queue
server** that survives restarts — not in the agent, not in the panels. Adding
a plan to the queue and starting the queue are **two separate, deliberate
steps**, and only starting is guarded. **Stopping is never locked** — no
token, no switch, no state can take the stop and abort buttons away.

.. raw:: html
   :file: ../../_diagrams/bluesky-queue.html

Everything below is the same queue seen from three sides — pick the one you
work in.

One queue, three ways to drive it
=================================

.. tab-set::

   .. tab-item:: Panels

      The **PLAN** tab composes; the **BLUESKY** tab runs and watches.

      - **Add to queue** (PLAN) — queues exactly the draft on your screen,
        after a confirming click. Nothing moves.
      - **Start queue** (BLUESKY) — one click; drains every queued item, in
        order.
      - **Stop after current item** and **Abort running plan** (BLUESKY) —
        always clickable, in every state. Abort asks for a second,
        confirming click.
      - Every queued row has reorder (↑ ↓) and remove (✕) buttons — the queue
        is editable right up until it runs.
      - A **Simple mode** hides the expert details and leaves the essentials:
        the form, the queue, the results, and the halts.

      See :doc:`/how-to/web-terminal/panels` for where these tabs live.

   .. tab-item:: Chat with the agent

      Ask in plain language — "queue that plan and start it", "stop the
      queue", "abort the plan" — and the agent drives the same queue with a
      small set of tools:

      - ``get_draft`` / ``set_draft`` — compose the shared draft you see in
        BLUESKY's Plans view.
      - ``queue_add`` / ``queue_start`` — the two steps. Both ask for your
        approval, and both are switched off entirely while the project's
        control-system writes are disabled.
      - ``queue_stop`` / ``stop_run`` — the two halts. Never switched off.
      - ``queue_list`` / ``queue_status`` / ``list_runs`` / ``get_run_data``
        — read what is queued, running, and measured.
      - ``get_run_figure`` — read the same figure the BLUESKY panel is
        drawing, so you and the agent are discussing one picture.

      A bundled skill (``operating-bluesky-plans``) teaches the agent this
      flow, so you rarely need to name a tool yourself.

   .. tab-item:: HTTP API

      The panels and the agent both talk to the **Bluesky bridge**, and your
      own tooling can too:

      .. code-block:: text

         POST /queue/items          add the current draft revision
         POST /queue/start          start draining (needs the launch token)
         POST /queue/stop           stop after the running item
         POST /queue/abort          abort the running plan — never gated
         GET  /queue                what is queued and running
         GET  /runs                 recent runs; /runs/<id>/data for the numbers,
                                    /runs/<id>/figure for the plotted view

      Every refusal comes back with a ``detail`` object of the form
      ``{"code": ..., "detail": ...}`` — a stable code for software to
      branch on, a sentence for a human to read. The panels and the agent
      show that same sentence, so every surface describes the same event
      the same way.

Managing the queue
==================

The queue is a plan of what the machine is about to do, and it stays
editable while idle: reorder items, remove them, keep adding. Two honest
quirks worth knowing:

- **Progress can be absent, and absent is not zero.** Plans that cannot
  predict their total point count report "N points so far" rather than a
  made-up percentage.
- **The run list is recent history, not the archive.** A run the list has
  forgotten still has its data — see *Where the data lives* below.

.. dropdown:: What needs arming — the full picture
   :color: info
   :icon: shield-check

   The **launch token** is a credential the deployment holds; it guards
   exactly the operations that send work toward hardware.

   .. list-table::
      :header-rows: 1
      :widths: 55 45

      * - Operation
        - What it needs
      * - Compose or edit the shared draft
        - Nothing — composing never touches hardware.
      * - Add to an **idle** queue
        - Nothing — the item just waits. (One exception: a queue server
          configured to start itself — never OSPREY's setup — makes every
          add an armed one.)
      * - Add while the queue is **running**
        - The launch token — this hands work straight to a moving machine.
      * - Start the queue
        - The launch token.
      * - Stop the queue / abort the running plan
        - Nothing. Ever. Anywhere.
      * - Withdraw a pending stop
        - The launch token — it lets the queue keep draining.

   The agent is held to a harder rule on top of this: while the project's
   ``control_system.writes_enabled`` switch is off, its ``queue_add`` and
   ``queue_start`` tools are denied outright — it cannot queue or start
   anything, even on an idle queue. Its halts and its read tools are never
   taken away.

   In a deployed control room the agent holds the launch token only where the
   deployment grants it — to a persona configured for control-system writes
   that also runs the bluesky MCP server. Where it is granted, the agent's
   ``queue_start`` arms the queue itself, and your approval of that tool call
   is the arming decision. Where it is not, ``queue_start`` is refused with
   ``launch_token_required`` and the start stays with you, from the BLUESKY
   queue panel's own **Start queue** button.

.. dropdown:: When something is refused
   :color: info
   :icon: alert

   A refusal always says why. The codes you will actually meet:

   ``stale_draft_revision`` / ``draft_revision_already_launched``
      The draft changed since you looked, or that exact revision has already
      been queued once. Re-read the draft; edit it to mint a fresh revision
      for a repeat.

   ``launch_token_required``
      The operation was armed and the caller held no valid token. Nothing was
      started. An agent meets this where the deployment did not grant it a
      token, where the token it holds does not match the bridge's, or where
      no launch token is configured at all — hand the start to the operator.

   ``browse_only_connector``
      This deployment cannot execute plans at all — it is pointed at the
      ``mock`` control system. Composing still works; the refusal names the
      command that switches to an executing connector.

   ``session_plan_unvalidated``
      An agent-written plan must pass validation, byte for byte, before it
      runs — see :doc:`write-plans`.

   ``interrupted_item_in_queue``
      The queue still holds a plan someone stopped — see the next dropdown.

   ``manager_unreachable``
      The queue server is not answering — often it is simply still starting.
      Wait a moment and retry.

.. dropdown:: After an emergency stop
   :color: info
   :icon: stop

   A stopped plan does not vanish. The queue server records the run in
   history **and puts a copy of the item back at the front of the queue**, so
   a human can decide what happens next. Until that copy is removed, every
   attempt to start the queue is refused — a plan someone emergency-stopped
   can never sneak back onto the machine.

   Removing it is the deliberate step: the ✕ on its queue row. To actually
   run it again afterwards, stage it through the draft and add it afresh.

.. dropdown:: Where the data lives
   :color: info
   :icon: database

   - **While a plan runs**, the panels and the agent read live rows from the
     bridge's own buffer.
   - **After that**, the data is durable in **Tiled**, the deployment's data
     store (part of the tutorial preset; optional elsewhere) — it outlives
     the run list, the queue server's history, and any restart. A run id
     whose entry has aged out of the list still answers with its data.
   - **The queue itself** lives in the queue server's own storage, so a
     bridge restart changes nothing about what is queued. The one gap: live
     rows of a run that is happening *during* a bridge restart are missing
     from the live view until the next run starts — the run itself keeps
     going and its data still lands in Tiled.

.. dropdown:: For deployers — what is running
   :color: info
   :icon: server

   A project built from the ``control-assistant`` preset brings the whole
   stack up with ``osprey up``: the **bridge** (the HTTP front door,
   port 8090), the **queue server** with its own storage, the **bluesky-web**
   sidecar serving the BLUESKY panel (port 8095), the **Virtual Accelerator** (the
   preset's default control system), and — when enabled — **Tiled** (port
   8091). The launch token is minted automatically at deploy time and stored
   in the project's ``.env``.

   The build profile's ``bluesky:`` block sets the ports, the Tiled store,
   which plans the catalog carries, and the device file plans may drive or
   record — see :doc:`/reference/configuration/profile`.

   Whether a deployment can execute plans at all is decided by its control
   system: ``virtual_accelerator`` and ``epics`` can, ``mock`` is
   browse-only. The panels and the agent both surface this as a capability
   banner; on a browse-only deployment it names the flip:

   .. code-block:: bash

      osprey set connector=virtual_accelerator

   Run ``osprey build`` and ``osprey up`` afterwards to carry the change into
   the running stack. That switch needs a real archive behind it: a deployment
   created from the ``control-assistant`` preset has one and the flip just
   works; one still reading the mock archiver is refused at build time, and told
   to point ``archiver.type`` at a store its deployment writes first — see
   :doc:`../control-systems/use-virtual-accelerator`.

   One timing detail worth knowing: the channel limits a plan's writes are
   checked against come from the file
   ``control_system.limits_checking.database_path`` names, and ``osprey build``
   stages its **own copy** of that file for the plan lane. So widening or
   tightening a limit in your deployment repository reaches the queue server at
   the next ``osprey build`` (and ``osprey up``) — not the moment you save the
   file.

.. seealso::

   :doc:`run-first-plan`
      The worked example, from asking to watching points land.

   :doc:`write-plans`
      Trust tiers and adding plans of your own.

   :doc:`/how-to/control-systems/use-virtual-accelerator`
      The connector that makes a deployment able to execute.
