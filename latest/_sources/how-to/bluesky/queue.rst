===================
Scans and the Queue
===================

Three sentences carry everything on this page. Scans live in a **queue
server** that survives restarts — not in the agent, not in the panels. Adding
a scan to the queue and starting the queue are **two separate, deliberate
steps**, and only starting is guarded. **Stopping is never locked** — no
token, no switch, no state can take the stop and abort buttons away.

.. mermaid::

   flowchart LR
       A["Shared draft<br/>compose & review"] -->|add| B["Queue<br/>durable, ordered"]
       B -->|"start (armed)"| C["Machine<br/>one scan at a time"]
       C -.->|"stop / abort — always available"| B

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

      Ask in plain language — "queue that scan and start it", "stop the
      queue", "abort the scan" — and the agent drives the same queue with a
      small set of tools:

      - ``get_draft`` / ``set_draft`` — compose the shared draft you see in
        BLUESKY's Plans view.
      - ``queue_add`` / ``queue_start`` — the two steps. Both ask for your
        approval, and both are switched off entirely while the project's
        control-system writes are disabled.
      - ``queue_stop`` / ``stop_run`` — the two halts. Never switched off.
      - ``queue_list`` / ``queue_status`` / ``list_runs`` / ``get_run_data``
        — read what is queued, running, and measured.

      A bundled skill (``operating-bluesky-scans``) teaches the agent this
      flow, so you rarely need to name a tool yourself.

   .. tab-item:: HTTP API

      The panels and the agent both talk to the **Bluesky bridge**, and your
      own tooling can too:

      .. code-block:: text

         POST /queue/items          add the current draft revision
         POST /queue/start          start draining (needs the launch token)
         POST /queue/start-request  ask a token holder to start — arms nothing
         POST /queue/stop           stop after the running item
         POST /queue/abort          abort the running plan — never gated
         GET  /queue                what is queued and running
         GET  /runs                 recent runs; /runs/<id>/data for the numbers

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
      * - Ask for a start (file a start request)
        - Nothing — the request arms nothing. Confirming it *is* the
          token-gated start, done from the queue panel.
      * - Stop the queue / abort the running plan
        - Nothing. Ever. Anywhere.
      * - Withdraw a pending stop
        - The launch token — it lets the queue keep draining.

   The agent is held to a harder rule on top of this: while the project's
   ``control_system.writes_enabled`` switch is off, its ``queue_add`` and
   ``queue_start`` tools are denied outright — it cannot queue or start
   anything, even on an idle queue. Its halts and its read tools are never
   taken away.

   In a deployed control room the agent's environment never holds the launch
   token at all — the token lives with the operator panels. The agent's
   ``queue_start`` then files a **start request**: it appears in the BLUESKY
   queue panel beside the queue it would drain, with *Confirm start* and
   *Dismiss* controls. Confirming fires the panel's own token-carrying start;
   dismissing starts nothing. Either way, the human's click is the arming
   decision.

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
      started. (An agent asking for a plain start never hits this — it files a
      start request for the panel instead.)

   ``queue_empty``
      A start was requested with nothing queued, so there was nothing a
      confirmation could run. Stage and add a plan first.

   ``browse_only_connector``
      This deployment cannot execute scans at all — it is pointed at the
      ``mock`` control system. Composing still works; the refusal names the
      command that switches to an executing connector.

   ``session_plan_unvalidated``
      An agent-written plan must pass validation, byte for byte, before it
      runs — see :doc:`write-plans`.

   ``interrupted_item_in_queue``
      The queue still holds a scan someone stopped — see the next dropdown.

   ``manager_unreachable``
      The queue server is not answering — often it is simply still starting.
      Wait a moment and retry.

.. dropdown:: After an emergency stop
   :color: info
   :icon: stop

   A stopped scan does not vanish. The queue server records the run in
   history **and puts a copy of the item back at the front of the queue**, so
   a human can decide what happens next. Until that copy is removed, every
   attempt to start the queue is refused — a scan someone emergency-stopped
   can never sneak back onto the machine.

   Removing it is the deliberate step: the ✕ on its queue row. To actually
   run it again afterwards, stage it through the draft and add it afresh.

.. dropdown:: Where the data lives
   :color: info
   :icon: database

   - **While a scan runs**, the panels and the agent read live rows from the
     bridge's own buffer.
   - **After that**, the data is durable in **Tiled**, the deployment's data
     store (part of the tutorial preset; optional elsewhere) — it outlives
     the run list, the queue server's history, and any restart. A run id
     whose entry has aged out of the list still answers with its data.
   - **The queue itself** lives in the queue server's own storage, so a
     bridge restart changes nothing about what is queued. The one gap: live
     rows of a scan that is running *during* a bridge restart are missing
     from the live view until the next scan starts — the scan itself keeps
     going and its data still lands in Tiled.

.. dropdown:: For deployers — what is running, and the config block
   :color: info
   :icon: server

   A project built from the ``control-assistant`` preset brings the whole
   stack up with ``osprey deploy up``: the **bridge** (the HTTP front door,
   port 8090), the **queue server** with its own storage, the **PLAN /
   BLUESKY panels** sidecar (port 8095), the **Virtual Accelerator** (the
   preset's default control system), and — when enabled — **Tiled** (port
   8091). The launch token is minted automatically at deploy time and stored
   in the project's ``.env``.

   The build profile's ``bluesky:`` block accepts exactly five keys — a
   misspelled or unknown key **fails the build** and prints the valid set:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Key
        - What it does
      * - ``port``
        - The bridge's port (default 8090).
      * - ``tiled_enabled``
        - Deploy the Tiled data store alongside the stack.
      * - ``tiled_port``
        - Tiled's port (default 8091).
      * - ``plan_dir``
        - A directory of your facility's own scan plans — see
          :doc:`write-plans`.
      * - ``excluded_plans``
        - Plans to remove from the catalog entirely, e.g. ``[orm]``.

   Whether a deployment can execute scans at all is decided by its control
   system: ``virtual_accelerator`` and ``epics`` can, ``mock`` is
   browse-only. The panels and the agent both surface this as a capability
   banner; on a browse-only deployment it names the flip:

   .. code-block:: bash

      osprey config set-control-system virtual_accelerator

   That switch needs a real archive behind it. A project built from the
   ``control-assistant`` preset has one and the command just works; a project
   still reading the mock archiver is refused, and told to point
   ``archiver.type`` at a store its deployment writes first — see
   :doc:`../use-virtual-accelerator`.

.. seealso::

   :doc:`run-first-scan`
      The worked example, from asking to watching points land.

   :doc:`write-plans`
      Trust tiers and adding plans of your own.

   :doc:`/how-to/use-virtual-accelerator`
      The connector that makes a deployment able to execute.
