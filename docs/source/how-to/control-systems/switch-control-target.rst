.. _how-to-switch-control-target:

=====================================
Switch the Control Target at Run Time
=====================================

How to rehearse a piece of work on the **virtual accelerator** and then run the
same work on the **live machine**, without rebuilding the project or restarting
anything.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - The rehearse-then-run workflow, and the two tools it uses
   - What a deployment needs before either target can be selected
   - How to read the target roster, including what its reachability rows can
     and cannot prove
   - What the switch refuses, and what each refusal is asking you to do
   - How you can tell, at every step, which machine a call is about
   - What happens to Bluesky plans while a session is switched

   **Prerequisites:** a deployment that describes both a real control system and
   a virtual accelerator (see `What a deployment needs first`_), and
   :doc:`use-virtual-accelerator` for the simulator itself.

The workflow
============

The point of the switch is a rehearsal. You have a script, a set of writes or a
plan you would rather not try for the first time on the real machine, so you run
it against the simulator, look at what it did, and then run it for real —
in one session, with the same tools, the same limits and the same approval
prompts on both.

Two tools do this:

``control_target``
   Reports where the session is pointed and where else it could go. Read-only:
   it opens no connections and changes nothing, so it is safe to ask at any
   moment, including before anything has ever been switched.

``control_target_set(target)``
   Moves the session. ``target`` is either ``va`` (the virtual accelerator) or
   ``live`` (the real machine this deployment describes). It asks for your
   approval first, and the prompt names the machine you would be moving to.

Ask the OSPREY agent for these in plain language — *"what am I pointed at?"*,
*"switch to the virtual accelerator"*. A session on the simulator looks like a
session on the machine: reads, writes and the python executor all follow the
switch. The archive does not — archive reads keep the deployment's one
configured archiver, and are stamped with the session target and the archiver
that served them, so which machine a set of history is about is never
ambiguous.

Rehearse on the virtual accelerator
-----------------------------------

Start by asking where you are, then move to the simulator::

   > what control target is this session on?
   > switch to the virtual accelerator

The switch starts the new connection **before** it retires the old one, and
proves it by reading one channel — the target's ``probe_channel`` — through the
new connection. Only when that read answers does the session move. A switch
that cannot prove the destination leaves you exactly where you were, still
working, with the reason reported.

Now do the work: run the script, make the writes, look at the results. Nothing
about the tools changes; only the machine behind them does.

Go live
-------

When the rehearsal looks right::

   > switch to the live machine

Moving **toward** the live machine is the one direction with an extra gate.
Two things must already be true of the deployment:

**A strict limits posture.** ``control_system.limits_checking.enabled`` must be
``true`` and ``control_system.limits_checking.allow_unlisted_channels`` must be
``false`` — that is, every writable channel is on the list, and a channel that
is not on the list is refused rather than allowed through.

**An operator acknowledgment.** ``control_system.target_switch.live_gateway_acknowledged``
must be set, to the hostname of the live gateway this deployment is configured
against. Setting it is you saying *"the gateways in this config really are my
facility's machine"*. Nothing infers that: the shipped example value looks like
a real hostname, so no check could tell an operator's answer from a placeholder.
The key ships commented out, and a deployment that has not answered cannot
switch to live.

There is one exemption. If the live machine **is** this deployment's baseline —
the target it was built for — a session that has wandered off to the simulator
can always come home, with neither the posture nor the acknowledgment. Stranding
a session on a simulator when the operator asked for the real machine is the
less safe outcome of the two.

Coming home
-----------

Nothing needs to be undone. The session target is not saved anywhere that
outlives the session: every time the controls server starts it returns to the
deployment baseline and clears what the previous session left behind. Closing
the session is enough.

What a deployment needs first
=============================

A project can only switch if its config describes **both** targets. That is a
build-time property, and ``osprey build`` renders both connector blocks for you:
a project generated from the standard template gets a ``virtual_accelerator:``
block beside its ``epics:`` one, with the simulator's gateway pointed at the
Virtual Accelerator the stack deploys.

One key is not filled in for you. Each target names a ``probe_channel`` — the
channel the switch reads to prove that target is reachable:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Target
     - ``probe_channel``
   * - ``va``
     - Shipped set, to a placeholder you replace with a channel your virtual
       accelerator's model actually serves.
   * - ``live``
     - Shipped **commented out**. A facility's channel names cannot be guessed,
       and a placeholder here would make the live target look ready while naming
       a channel nothing answers.

A target with no ``probe_channel`` is never switched to, and the roster says so
by name. Naming the live machine's probe channel is therefore a deliberate act
by whoever knows the facility — the same posture as the acknowledgment key
above.

Two more keys bound the switch itself, both under
``control_system.target_switch:``: ``drain_timeout_s`` (default 5) is how long
work already in flight gets to finish on the old target before it is torn down
regardless, and ``probe_interval_s`` (default 30) is how often the background
reachability check runs.

.. note::

   Changing the target is **not** a config change and not a rebuild.
   ``control_system.type`` only sets the target a session *starts* on. The
   agent's own setup guidance carries the same rule in its hot/cold settings
   table: target changes go through ``control_target_set``, and nobody should be
   sent to ``osprey build`` to change which machine a session is talking to.

Reading the roster
==================

``control_target`` answers with one row per target. The row says what the target
*is*, whether the session may move there **right now**, and — where something
has actually measured it — whether its gateway answered.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - What the row carries
     - What it means
   * - ``available_now`` and ``reason``
     - Whether this session may switch there at this moment, and if not, why
       not. The reason is the same word the switch itself would refuse with, so
       the two can never tell you different stories.
   * - ``eligible_from_baseline``
     - The same question asked as if the session sat on the deployment baseline
       — the static view, unaffected by where you happen to be. On a
       live-baseline deployment with no acknowledgment set, whose session is
       currently on the simulator, this reads ``false`` for ``live`` while
       ``available_now`` reads ``true``; that is the coming-home exemption made
       visible, not a contradiction.
   * - ``endpoints``
     - One entry per configured gateway role — ``read_only``, ``write_access``,
       and ``pva`` where it exists — with the host, port and routing mode
       derived from config, plus ``selected_role``: the gateway this deployment
       would actually use, which follows the write posture.
   * - ``endpoint_tcp`` and ``probed_at``
     - The background check's last observation of that gateway, and when it made
       it. A row older than three probe intervals is marked ``stale`` rather
       than presented as current.
   * - ``writes_permitted``, ``real_machine``, ``probe_channel``
     - Whether writes are possible at all on this deployment, whether this
       target is the real machine, and the channel a switch would prove it with.

Two things the roster is careful about are worth knowing, because they are the
difference between a report you can act on and one that flatters you:

**It is correct before anything has been switched.** A target nobody has ever
activated is judged from configuration alone, so the roster is useful as a
first question rather than only as an after-the-fact record.

**A reachability row is not offered where nothing measured one.** If no
measurement exists, the row simply carries no ``endpoint_tcp`` — "not measured"
and "measured as down" are different claims and are never spelled the same way.

.. important::

   **The honest limit of the reachability check.** A gateway reached by
   *address list* rather than by name server reports ``not_applicable``, not a
   status. Channel Access — the EPICS protocol OSPREY speaks — finds channels by
   UDP broadcast in that mode, and a TCP probe proves only that something is
   listening on a socket, not that a channel search would be answered. Reporting
   it as "up" would be a guess dressed as a measurement.

   On a stock EPICS deployment that is exactly the live machine's situation: it
   has no continuous liveness row, and the probe the switch performs — a real
   read of a real channel, through the connection it is about to hand you — is
   its only real evidence. That probe is why the switch is trustworthy even
   where the background check has nothing to say.

What the switch refuses, and why
================================

Every refusal names one thing you can act on, and refusals are reported in a
fixed order from "this session may never switch" to "this destination is not
usable right now".

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - What you will see
     - What it means
   * - **This run is read-only**
     - The session was started in read-only mode, which is a claim about the
       whole run. A switch changes session state, so read-only sessions stay on
       the deployment baseline. Re-run without read-only mode.
   * - **An execution is in flight**
     - A python execution is running, and it was launched against the target it
       started on. Wait for it to finish or stop it, then switch. The refusal
       names which target the running work is on, and whether it belongs to this
       session or another one sharing the deployment.
   * - **Already there**
     - The session is on that target already. The active target always answers
       this, whatever else would also be true of it.
   * - **The target is not configured**
     - No connector block for that target, no gateways table, or no entry for
       the gateway role this deployment would select. This is a build or config
       gap, not something the session can resolve.
   * - **No probe channel**
     - The target names no ``probe_channel``, so nothing could prove it
       reachable. For the live machine this is the shipped state — see
       `What a deployment needs first`_.
   * - **A simulated present with an invented past**
     - Switching to the virtual accelerator is refused while the deployment's
       archiver is the mock one, which makes history up at read time. The pairing
       would put a made-up past next to a modelled present with nothing linking
       them. See "The honesty rule" in :doc:`use-virtual-accelerator`.
   * - **The live posture is not in place**
     - Strict limits, or the operator acknowledgment, or both. See `Go live`_.

Two more refusals arrive *after* a switch rather than instead of one, and both
exist for the same reason: something was approved for one machine and must not
quietly land on another.

- **A write approved before the switch.** Every write carries the target and
  the generation it was approved under — a counter every switch advances, so
  even a round trip back to the same target retires older approvals — and
  refuses if either moved before it executed. Nothing is written; ask for the
  write again on the target you actually mean.
- **A python execution that outlived the switch.** A running script keeps the
  target it was launched with, and its writes refuse once the session moves past
  that point rather than being redirected.

Finally, two surfaces stay deliberately pinned to the deployment baseline while
a session is switched, and say so rather than following along: driving a Phoebus
widget is refused, and ``osprey health`` keeps reporting against the baseline
with an added line naming both targets. Both talk to the deployment's own
configured stack, which the session's choice does not move.

Failures name the machine they happened on
==========================================

On a deployment with two targets, "the read of ``SR:...:RB`` timed out" is a
materially different situation on the live machine than on the simulator. So
every control-system failure envelope — a connect failure, a timeout, a limits
refusal, a write the control system itself denied — names the target the
session was pointed at when it failed: a human clause in the message
(``active target: LIVE MACHINE at 10.0.0.5:5064``) and a machine-readable
``details.active_target`` block carrying the target name, its label, and the
endpoint where the configuration knows one. The agent narrating a failure, and
any script asserting on one, can attribute it to the right machine from the
payload alone instead of reconstructing the answer from session memory.

Knowing which machine you are on
================================

The switch would be a hazard if it were quiet. It is not:

- **Every approval prompt names the target**, not only the write prompts — a
  queue start, a patch or an execution all carry the line too, so its absence is
  never something you learn to read as safe. The live machine is named as
  ``LIVE MACHINE`` with the gateway the session actually holds; the simulator as
  ``virtual accelerator (simulation)``. If the target cannot be read at all, the
  line says so explicitly instead of disappearing.
- **Results and artifacts are stamped.** Archive reads carry the session's
  target and the archiver that served them, so a saved plot still says what it
  is about a week later.
- **The session's target is visible in the Web Terminal** activity stream as
  work happens.
- **Nothing survives the session.** Every controls-server start returns to the
  deployment baseline. There is no saved preference that could quietly point a
  later session at the real machine.

Bluesky plans while switched
============================

A Bluesky **plan lane** is a whole plan stack — bridge, queue manager, worker —
wired at build time to one target. Every deployment renders one.

On a single-lane deployment, which is every deployment by default, queueing or
starting a plan is **refused** while the session is pointed somewhere the lane
does not serve. The refusal says which target the lane serves, and that adding a
second lane is a deployment change rather than something to retry.

Setting ``bluesky.second_lane: true`` in the build profile renders a second
complete lane, one per target. The switch then stops being a refusal and becomes
an address: a plan is routed to the lane serving the session's target, ``queue_add``
reports the lane it bound the plan to, and ``queue_start`` must name that lane —
so a plan composed for the simulator cannot be started on the machine because
the session moved in between. See :doc:`../bluesky/index` for the plan stack
itself.

See the whole loop run
======================

The repository ships a script that drives one session across two Channel Access
endpoints and prints an audit trail you read afterwards:

.. code-block:: bash

   PYTHONPATH=src:packages/osprey-connectors/src \
     python scripts/demo_target_switch.py --self-provision

``--self-provision`` stands the posture up locally: two virtual-accelerator
containers on two ports, one of them seeded so the two serve *different* values
for the same channel. The trail then shows, step by step, that the virtual
accelerator is eligible before any switch has happened, that the probe channel
reads one value before the switch and a different one after, that a python
execution follows the session into its own subprocess, and that a write made on
one target is not visible on the other. ``--config <path>`` runs the same
narrative against a deployment you already have, and ``--dry-run`` prints the
plan without touching anything.

.. seealso::

   - :doc:`use-virtual-accelerator` — the simulator itself, the archive it
     deploys, and the one configuration the stack refuses.
   - :doc:`/architecture/python-executor` — how executions are launched, and what the
     target stamp pins them to.
   - :doc:`../bluesky/index` — the plan stack that plan lanes belong to.
