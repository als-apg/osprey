============================
Run Scans Through the Queue
============================

How scans actually execute: a queue server holds them, adding and starting are
two separate steps, and the halts are always available. This is the reference
behind the walkthrough in :doc:`agent-assisted-grid-scan`.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Why running a scan is two steps, and what each one costs
   - Which operations are armed and which are free
   - How a session-tier plan becomes runnable
   - How to read a refusal
   - The difference between stopping the queue and aborting a scan, and what
     to do with the item an abort leaves behind

   **Prerequisites:** a project with the bluesky scan stack deployed (see
   :doc:`/getting-started/control-assistant`).

Overview
========

Scans do not run inside the Bluesky bridge. They run in a **queue server** — a
separate worker process with its own copy of the devices — and the bridge is
the API in front of it. That is what makes a queue durable: it survives a
bridge restart, because the queue lives with the queue server, not in the
bridge's memory.

Everything below has three faces, and they are the same operations:

- **the panels** — PLAN composes, BLUESKY runs and watches
  (:doc:`web-terminal/panels`);
- **the OSPREY agent** — ``queue_add``, ``queue_start``, ``queue_stop``,
  ``stop_run``, and the read tools ``queue_list`` / ``queue_status``;
- **the bridge's HTTP API** — ``POST /queue/items``, ``POST /queue/start``,
  and so on, for anything you integrate yourself.

Can this deployment run a scan?
===============================

Ask before composing anything. The bridge reports a **capability** record —
``{can_execute, reason, detail}`` — alongside its health; the panels show it as
a banner and the agent reads it with ``queue_status()``.

When ``can_execute`` is false the deployment cannot run plans, and the queue
**refuses to hold items** rather than accepting work it could never drain — so
you find out while composing, not after. ``reason`` is a short code
(``browse_only_connector``, ``unsupported_connector``, ``config_unreadable``,
``manager_not_configured``, ``manager_unreachable``) and ``detail`` is a
sentence written for a human, which for a browse-only deployment names the
exact command that flips it.

A browse-only deployment is still useful: plans can be listed, authored,
validated and staged into the shared draft. The ``mock`` connector is the
common case — see :doc:`use-virtual-accelerator`.

Running is two steps: add, then start
=====================================

**Step 1 — add.** The PLAN panel's **Add to queue** (agent: ``queue_add``)
puts the plan in the queue and stops there. It queues the exact draft revision
on screen, pinned, so a draft that has moved on since is refused rather than
run by surprise, and a revision can be queued only once — a repeat run needs a
fresh edit to mint a new revision.

**Step 2 — start.** The BLUESKY panel's **Start queue** (agent:
``queue_start``) is what begins motion. It drains **the whole queue, in
order** — not only the item you just added — so read the list first. Nothing
else ever starts the queue: the queue server's own autostart stays off, so
every start is a deliberate act by a human or the agent.

The split is the point. It lets a queue be assembled and reviewed while
nothing moves, and it puts the arming gate on the step that actually starts
hardware.

What is armed, and what is not
==============================

What the bridge asks for, per operation:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Operation
     - What it needs
   * - Editing the shared draft
     - Nothing. Composing never touches hardware.
   * - Add to an **idle** queue
     - Nothing. The item sits there; nothing drains it.
   * - Add to a queue that is **already running**
     - The launch token. Adding here hands the item straight to hardware.
   * - Start the queue
     - The launch token.
   * - Stop the queue / abort the running plan
     - Nothing, ever. See `Stopping: two different halts`_.
   * - Withdraw a pending stop
     - The launch token — it lets the queue resume toward hardware.

The bridge is what decides whether a particular add was armed, because only it
knows the queue server's live state under a lock.

For the OSPREY agent, ``control_system.writes_enabled: false`` is a harder stop
than that table suggests. Its ``queue_add`` and ``queue_start`` are denied
outright: the build writes them into the project's deny list, and a pre-tool
hook refuses them again at call time. So with writes off the agent cannot queue
or start anything, on an idle queue or otherwise. (Its tools *also* withhold
the launch token in that state — a backstop for any consumer running without
those hooks, not the layer doing the work.) What the agent keeps is everything
short of the queue: listing, authoring, validating, and staging plans into the
shared draft.

The panel is not gated that way. It can still add to an idle queue on a
writes-disabled deployment, because that add moves nothing and the start it
would need is separately token-gated.

Halting is never withheld from either. The kill switch selects the arming pair
by name, deliberately leaving stop and abort reachable — see `Stopping: two
different halts`_.

Session plans have to be validated first
========================================

A plan that ships with OSPREY, with a preset, or with your facility reaches
the queue server's worker on its own. A **session plan** — one the agent
authors during a conversation with ``write_plan`` — does not, and it is not
trusted the way the others are: it is trusted only for the exact bytes that
passed validation. Three things happen, in this order:

1. **Validate.** ``validate_plan`` runs the bridge's validator against the
   file's current content, and records the pass against that content's hash.
2. **Upload.** On a pass, and only on a pass, those bytes are loaded into the
   queue server worker's namespace. The response says whether the upload
   landed; a pass with no upload is still a genuine pass, but the plan cannot
   be queued until one does.
3. **Re-check, at add and at start.** The plan is admitted only if all three
   still hold: the file's *current* hash has a passing record, that same hash
   is what was uploaded, and the worker still lists the plan. Any gap is a
   refusal telling you to re-validate — never a silent re-upload.

So editing the file drops it back to unvalidated, because the hash changed. So
does restarting the bridge, which forgets its validation records while the
queue itself survives: a session-plan item that outlived a restart is refused
until it is validated again. Both are the same rule — what runs must be bytes
a validation actually passed.

Refusals say why, in a way software can read
============================================

Every refusal comes back as an HTTP error whose body carries
``{"code": ..., "detail": ..., ...}``: ``code`` is a stable machine-readable
string to branch on, ``detail`` is the sentence to show a human. The panels
and the agent both branch on the code and relay the sentence unchanged, so
everyone describes the same event the same way.

The ones worth recognizing:

``launch_token_required``
   The operation was armed and the token was missing or wrong. For an add,
   ``manager_state`` says what made it armed. If ``item_left_behind`` is true,
   an item could not be withdrawn and is sitting in a running queue —
   ``item_uid`` names it, and a human has to deal with it.

``stale_draft_revision`` / ``draft_revision_already_launched``
   The draft moved on, or that revision is already queued. Re-read the draft
   and add the current revision; for a repeat run, edit the draft to mint a
   new one.

``session_plan_unvalidated`` / ``session_plan_not_in_namespace``
   See the section above. At start, one stale plan refuses the whole start.

``browse_only_connector`` / ``unsupported_connector`` / ``config_unreadable``
   This deployment cannot run plans. ``capability.detail`` names the flip.

``manager_not_configured`` / ``manager_unreachable`` / ``environment_unavailable``
   The queue server or its worker is not available. ``manager_unreachable`` can
   simply mean the stack is still starting, so a short retry is reasonable.

``interrupted_item_in_queue``
   A start was refused because the queue still holds a plan that was stopped —
   see `After an abort, the item is still in the queue`_.

Stopping: two different halts
=============================

They are not interchangeable, and the difference matters most in the moment
someone asks for one.

**Stop after current item** (agent: ``queue_stop``) lets the running scan
finish and stops the queue there. It does not touch a plan that is already
moving hardware.

**Abort running plan** (agent: ``stop_run``) stops the scan that is running
right now. Its cost is real and worth saying out loud before you press it: the
remaining points are discarded, the data already collected is kept, and **the
hardware is left wherever the scan had moved it** — an abort returns nothing to
a starting position.

Both are **completely ungated**, on every surface: no launch token, no writes
switch, no capability check. A halt that can fail to be available is not a
halt, so the abort stays reachable even on a deployment whose writes are
switched off. The panel's Abort takes a second, confirming click; that is the
only thing between you and it.

An abort is honest about what it achieved. ``abort_pending: true`` means the
queue server accepted it and is still unwinding — that is the abort landing,
not failing. A ``nothing_running`` refusal means there was nothing to stop. An
``abort_pause_timeout`` means **nothing was aborted and the plan may still be
running**; it is never dressed up as a halt.

After an abort, the item is still in the queue
==============================================

The queue server does not discard a plan it interrupted. It records the run in
history *and* pushes a copy of the item back to the **front of the queue**, so
an operator can decide what to do with it. This is not configurable, and it
applies to a plan that failed on its own just as much as to one a human
aborted.

OSPREY handles the two consequences rather than hiding them:

- The run reports as **stopped**, not pending — a plan someone stopped never
  reads as work still to come. (A plan that failed reports as **error**.)
- **Start queue is refused** while that copy is queued
  (``interrupted_item_in_queue``), so a scan someone emergency-stopped cannot
  go back on the hardware without a fresh decision.

The gate re-reads the queue on every start, so it refuses every start until
the item is gone. **Removing that item is the only way to unblock the queue** —
the ✕ on the queue row, or ``DELETE /queue/items/<uid>``. If you do want to run
the plan again, remove it first, and then stage it through the draft and add
it afresh as a second, deliberate step.

Watching a run
==============

The BLUESKY panel and the agent's read tools see the same thing. A run's
status is one of ``pending``, ``running``, ``completed``, ``stopped`` (a human
stopped it, by any route) or ``error``.

Two things read oddly at first and are both correct:

- **Progress can be absent**, and absent is not zero. A total point count can
  only be derived for plan shapes OSPREY recognizes, which agent-authored
  plans often are not — so the honest answer is "N points so far", never 0%.
- **A run id the run list has forgotten may still have data.** The list covers
  what the queue server still remembers; the data itself is durable in Tiled
  long after that.

One more, if you restart the bridge while a scan is running: its live rows are
gone until the run's *next* start document, because the live buffer is the
bridge's own. The scan keeps going and its data still lands in Tiled — only
the live view has a gap.

Items queued outside OSPREY carry no OSPREY run id and are absent from the run
list entirely. The queue view is the complete picture of what the machine is
about to do.

Coming from an earlier release
==============================

.. note::

   **Direct execution is gone.** Plans used to run in the bridge process, via
   ``POST /runs/{id}/launch`` and ``POST /draft/run``. Both routes — and
   ``POST /runs`` and ``POST /runs/{id}/stop`` alongside them — now answer
   ``410 Gone`` with a ``use_the_queue`` refusal naming their replacement,
   rather than 404ing and leaving a caller to guess where the capability went.
   The queue is the only path to hardware. If you integrated against those
   routes, replace launch with ``POST /queue/items`` followed by
   ``POST /queue/start``, and per-run stop with ``POST /queue/stop`` or
   ``POST /queue/abort``.

.. note::

   **The ``bluesky.demo_runner`` build-profile knob is gone**, along with the
   in-bridge runner it switched on. Remove it from any profile that still sets
   it.

   Unknown keys in a profile's ``bluesky:`` block now **fail the build** with
   the list of valid keys, where they used to be dropped in silence. That is
   what makes the removal above announce itself — a stale ``demo_runner:``
   now stops the build instead of vanishing — and it means a typo anywhere in
   that block (``tiled_enabld``, ``plan_dirs``) is caught at build time. The
   keys the block accepts are ``excluded_plans``, ``plan_dir``, ``port``,
   ``tiled_enabled`` and ``tiled_port``.

.. seealso::

   :doc:`agent-assisted-grid-scan`
      A worked example, from asking for a scan to watching points land.

   :doc:`web-terminal/panels`
      The PLAN and BLUESKY tabs these operations run through.

   :doc:`use-virtual-accelerator`
      The connector that makes a deployment able to execute scans.
