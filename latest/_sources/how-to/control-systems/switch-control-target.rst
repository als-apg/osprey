.. _how-to-switch-control-target:

=====================================
Switch the Control Target at Run Time
=====================================

How to move a running session between the machines a deployment describes —
rehearse a piece of work on the **virtual accelerator**, run it on the **live
machine**, or rehearse the whole go-live procedure on a **stand-in** — without
rebuilding the project or restarting anything.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - The three control targets, and which machine each one names
   - The rehearse-then-run workflow, and the two tools it uses
   - What a deployment needs before a target can be selected
   - How to read the target roster, including what its reachability rows can
     and cannot prove
   - What the switch refuses, and what each refusal is asking you to do
   - How you can tell, at every step, which machine a call is about
   - What happens to Bluesky plans while a session is switched

   **Prerequisites:** a deployment that describes more than one machine (see
   `What a deployment needs first`_), and :doc:`use-virtual-accelerator` for the
   simulator itself.

The workflow
============

The point of the switch is a rehearsal. You have a script, a set of writes or a
plan you would rather not try for the first time on the real machine, so you run
it against the simulator, look at what it did, and then run it for real —
in one session, with the same tools, the same limits and the same approval
prompts on both.

Three machines
--------------

A **control target** names a machine, and OSPREY knows three of them. A
deployment has the ones its config describes, which is often two:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Target
     - The machine it names
   * - ``live``
     - The facility's own control system — the gateways written under
       ``control_system.connector.epics``. A write here moves real hardware.
       ``osprey build`` never rewrites that block, so ``live`` means the same
       machine on every deployment.
   * - ``va``
     - The virtual accelerator: a simulator this deployment stands up, serving
       the same channel names over real Channel Access. A write here moves
       nothing real.
   * - ``standin``
     - The **live stand-in**: a second soft IOC the deployment runs for itself,
       configured from its own ``control_system.connector.live_standin`` block.
       It is a machine of its own, not a relabelled ``live`` — and it is
       operated like the real one, which is what makes it worth rehearsing on.

Two tools move between them:

``control_target``
   Reports where the session is pointed and where else it could go — one row
   per target this deployment configures. Read-only: it opens no connections
   and changes nothing, so it is safe to ask at any moment, including before
   anything has ever been switched.

``control_target_set(target)``
   Moves the session. ``target`` is ``va``, ``live`` or ``standin``, spelled
   exactly. It asks for your approval first, and the prompt names the machine
   you would be moving to.

Ask the OSPREY agent for these in plain language — *"what am I pointed at?"*,
*"switch to the virtual accelerator"*. A session on the simulator looks like a
session on the machine: reads, writes and the python executor all follow the
switch.

In the Web Terminal you can also move the session yourself, without asking the
agent: the :ref:`control-target chip <web-terminal-session-posture>` in the
header lists every target and offers **Switch** on the ones this session could
move to. The switch it asks for is the same one — the same gate, the same
checks, the same refusal words — and a target the gate would refuse shows that
word where the button would be.

Whether writes are *allowed* follows it too. Write posture is per control
target, so a deployment can be armed on the simulator and read-only on the
machine — the same tool call that moved a setpoint before the switch is refused
after it, naming the target that refused. That is a property of the deployment's
config, not of the switch, and it is set up in
:doc:`use-virtual-accelerator`. Ask ``control_target`` if you want to know
before you move: every row carries its own ``writes_permitted``.

The archive does not follow the switch — archive reads keep the deployment's one
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

A deployment baselined on the stand-in
(``control_system.type: live_standin``) starts every session on ``standin``, so
going to the real machine is one tool call — the same call, and the same
prompts, as on a deployment that never had a stand-in at all.

Read the roster before you ask for it, so you know what will happen::

   > what control target is this session on, and where else could it go?

Each row carries ``available_now`` and, when that is ``false``, the ``reason``
the switch itself would refuse with. Clearing those reasons **is** the go-live
procedure. Then::

   > switch to the live machine

Moving toward the live machine is the direction with the extra gates. The switch
checks them in this order and reports the first one that fails, so the answer
names the nearest thing to fix rather than the whole list.

**The gateways** (``gateways_missing``).
``control_system.connector.epics.gateways`` names where the live machine is
reached. Like everything else about that machine it ships commented out — a
facility's gateways cannot be guessed, and shipped values would point a stock
deployment at hardware it never configured — so on a fresh deployment this is
the first gate, and the live target reads *not configured* until you author
it. Uncomment the block with your facility's gateway addresses.

**A probe channel** (``probe_channel_missing``).
``control_system.connector.epics.probe_channel`` names the channel the switch
reads to prove the machine answered before the session moves onto it. It ships
commented out — a facility's channel names cannot be guessed, and a placeholder
would make the live target look ready while naming a channel nothing answers.
Set it to a channel your facility actually serves.

**A strict limits posture** (``limits_posture``).
Limits checking must be ``enabled: true`` with ``allow_unlisted_channels:
false`` — every writable channel is on the list, and a channel that is not on
the list is refused rather than allowed through. The posture is the target's
own: ``control_system.connector.<type>.limits_checking`` when that target's
connector type states a block, and the deployment-wide
``control_system.limits_checking`` when it does not. The refusal names whichever
of the two answered, so you edit the line that decides rather than one it
overrides. This gate guards ``standin`` too: both are machines you meet hardware
behaviour on, and a rehearsal on a permissive posture rehearses the wrong
facility.

**An operator acknowledgment** (``operator_ack_missing``).
``control_system.target_switch.live_gateway_acknowledged`` must be set, to the
hostname of the live gateway this deployment is configured against. Setting it
is you saying *"the gateways in this config really are my facility's machine"*.
Nothing infers that: the shipped example value looks like a real hostname, so no
check could tell an operator's answer from a placeholder. It is the live
machine's gate alone — the stand-in's equivalent was said in the build profile,
by the line that stood the stand-in up.

**An archive that is not the stand-in's** (``archive_belongs_to_standin``).
A deployment that runs a stand-in *and* records its own archive store is
recording the stand-in, so the history in that store is the stand-in's.
Selecting the live machine would splice a real machine's readings onto a
stand-in's past in one store, with nothing afterwards able to tell them apart.
Clear it by stopping one of the two — take ``archiver_recorder`` out of the
deployment's services, or drop ``virtual_accelerator.live_standin`` from the
build profile — and rebuild. The archive belongs to the machine it records.

There is one exemption. A session can always come **home** to the deployment's
own baseline — whichever of the three that is — with neither the limits posture,
the acknowledgment, nor the archive gate applied. The probe still runs: a target
that cannot prove itself reachable is never switched to, in either direction.
Stranding a session away from the machine its deployment was built for is the
less safe outcome of the two.

Coming home
-----------

Nothing needs to be undone. The session target is not saved anywhere that
outlives the session: every time the controls server starts it returns to the
deployment baseline and clears what the previous session left behind. Closing
the session is enough.

What a deployment needs first
=============================

A session can only be moved to a target this config describes. Each target reads
its settings from a connector block of its own, and a target with no block is
not offered at all — it has no roster row, rather than a row saying a machine
nobody deployed is unavailable.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Target
     - Its connector block
   * - ``live``
     - ``control_system.connector.epics`` — the facility's own, and yours to
       write. The build never touches it.
   * - ``va``
     - ``control_system.connector.virtual_accelerator`` — rendered for you, with
       the simulator's gateway pointed at the Virtual Accelerator the stack
       deploys.
   * - ``standin``
     - ``control_system.connector.live_standin`` — derived, and only when the
       build profile asks for a stand-in. See `Rehearse on a stand-in live
       machine`_.

One key is not filled in for you. Each target names a ``probe_channel`` — the
channel the switch reads to prove that target is reachable:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Target
     - ``probe_channel``
   * - ``va``
     - Shipped set, to a placeholder you replace with a channel your virtual
       accelerator's model actually serves.
   * - ``live``
     - Shipped **commented out**. A facility's channel names cannot be guessed,
       and a placeholder here would make the live target look ready while naming
       a channel nothing answers.
   * - ``standin``
     - Copied from the simulator's. The stand-in is the same soft IOC over the
       same machine model, so the channel that proves one proves the other. A
       deployment whose simulator names none gets none here either.

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
     - Whether writes are permitted on **this target**, whether this target is
       the real machine, and the channel a switch would prove it with. Two rows
       of the same deployment can disagree about the first one: write posture is
       per target, so a simulator may be armed beside a live machine that is not.
   * - ``limits_strict``
     - Whether **this target** runs the strict limits posture — limits checking
       on and unlisted channels refused — which is what the ``limits_posture``
       gate above requires. Rows can disagree here too, but only from the
       deployment's config: a deployment can relax unlisted channels for its
       simulator alone. Unlike ``writes_permitted``, no chip toggle
       moves it. A target whose config states neither setting reads
       ``false``, because a deployment that has stated nothing has refused
       nothing.

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
   * - **The stand-in is not this deployment's**
     - ``standin`` is refused unless the endpoint its block selects really is
       the stand-in this deployment co-deploys, on that port over loopback.
       Anything else — a tunnel, a gateway someone repointed — is a machine in
       its own right, and it must not sit behind a soft label. Use ``live`` for
       the machine your facility runs.
   * - **A simulated present with an invented past**
     - Switching to a machine this deployment stands up for itself — the virtual
       accelerator or the stand-in — is refused while the deployment's archiver
       is the mock one, which makes history up at read time. The pairing would
       put a made-up past next to a modelled present with nothing linking them.
       See "The honesty rule" in :doc:`use-virtual-accelerator`.
   * - **The strict limits posture is not in place**
     - Required toward ``live`` and toward ``standin`` alike. See `Go live`_.
   * - **The live machine's gates**
     - The operator acknowledgment, or the archive that belongs to the stand-in.
       Both are the live machine's alone. See `Go live`_.

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
with an added line naming the targets. Both talk to the deployment's own
configured stack, which the session's choice does not move.

Failures name the machine they happened on
==========================================

On a deployment with more than one target, "the read of ``SR:...:RB`` timed out"
is a materially different situation on the live machine than on the simulator. So
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
  ``LIVE MACHINE`` with the gateway the session actually holds; the stand-in as
  ``LIVE MACHINE (stand-in)``; the simulator as
  ``virtual accelerator (simulation)``. If the target cannot be read at all, the
  line says so explicitly instead of disappearing. (The web terminal names the
  same machines by what they are --- *Real machine*, *Rehearsal*, *Simulator*,
  or the deployment's own configured names --- and keeps these technical labels
  on its tooltips.)
- **Results and artifacts are stamped.** Archive reads carry the session's
  target and the archiver that served them, so a saved plot still says what it
  is about a week later.
- **The session's target is visible in the Web Terminal** activity stream as
  work happens, and on the :ref:`control-target chip
  <web-terminal-session-posture>` in the header, which names the machine the
  session stands on and the write state on *that* machine ---
  ``● Simulator · writes on``, ``● Real machine · writes locked``. The write
  state is per machine, so a deployment can arm its simulator and leave the
  live machine read-only; the chip is where that shows. It catches up with a
  switch a few seconds after one is made, whoever made it, and falls back to
  the deployment's default target when none can be resolved for the session.
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
complete lane. Which machine each lane serves is **derived**, never authored:
the first lane serves the deployment's baseline, and the second covers the other
interesting machine — a ``live`` or ``standin`` baseline gets a ``va`` lane, and
a ``va`` baseline gets a ``live`` lane. A ``standin`` lane dials the co-deployed
soft IOC by name, so there is nothing for you to supply; a ``live`` lane always
dials the facility's own gateway through ``EPICS_CA_NAME_SERVERS``, and
``osprey up`` refuses to start rather than let that lane come up searching for
channels at nowhere.

The switch then stops being a refusal and becomes an address: a plan is routed
to the lane serving the session's target, ``queue_add`` reports the lane it
bound the plan to, and ``queue_start`` must name that lane — so a plan composed
for the simulator cannot be started on the machine because the session moved in
between. Switching a session moves which lane it addresses; it does not restart
any lane's containers. See :doc:`../bluesky/index` for the plan stack itself.

The **BLUESKY panel** follows the same two-lane deployment with a picker of its
own. A panel is shared by every session, so it cannot follow any one session's
target; instead it shows which machine it is pointed at in its status strip,
labelled by target, and switching lanes there reloads the panel onto the other
lane's bridge: its plans, shared draft, queue and results all move together, and
starting the queue uses that lane's own launch token. A single-lane deployment
shows no picker and nothing about its panel changes.

Rehearse on a stand-in live machine
===================================

A generated project describes a simulator and a facility gateway, but the
gateway may not exist yet, or not from this laptop — so the ``live`` half of
this page cannot be tried at all. A **live stand-in** gives you something to
rehearse the procedure on: a second soft IOC the deployment runs for itself,
deployed as a third control target of its own.

Setting ``virtual_accelerator.live_standin: true`` in the build profile deploys
it, and the ``control-assistant`` preset ships that line on. What the build
derives from it is the stand-in's own ``control_system.connector.live_standin``
block and nothing else — where the stand-in listens, and what proves it
reachable. The ``epics`` block beside it is untouched, so a facility already
pointed at its own control system can stand a rehearsal up next to it without
losing sight of its machine.

**Set the posture yourself.** The strict limits pair is the profile's to state,
in its ``config:`` block. The stand-in gets no block of its own — it is
hardware-shaped, so it keeps the deployment-wide pair the live machine runs
under, and only the simulator is relaxed:

.. code-block:: yaml

   config:
     control_system.limits_checking.enabled: true
     control_system.limits_checking.allow_unlisted_channels: false
     control_system.connector.virtual_accelerator.limits_checking.enabled: true
     control_system.connector.virtual_accelerator.limits_checking.allow_unlisted_channels: true

That first pair is what a switch to either real-machine target requires. Without
it ``control_target_set standin`` refuses with ``limits_posture``, which is the
right refusal: a rehearsal on a permissive posture rehearses a facility you do
not have. Writing the relaxation under ``virtual_accelerator`` instead of
deployment-wide is what keeps the rehearsal honest — a per-type block replaces
the pair above for that type alone, and must state both settings or ``osprey
build`` and ``osprey validate`` refuse it.

**Three rows on the roster.** Ask where the session could go and you get one row
per configured machine::

   > what control target is this session on, and where else could it go?
   > switch to the stand-in

``control_target_set standin`` moves the session the same way every other switch
does — the probe, the approval prompt, the write posture that follows the
target. From there, ``control_target_set live`` walks the real go-live path
(`Go live`_) against a machine that cannot move a magnet.

**Start there rather than switch there.** ``osprey set connector=live_standin``
writes ``control_system.type: live_standin`` into the profile, and from the next
``osprey build`` every session *starts* on the stand-in — the posture for a
deployment that is not yet wired to its facility, or whose machine is down.

Flipping it back is three settings, not one, because the archive goes with the
machine. The ``va_archiver:`` block records *this* deployment's history from
the stand-in, and the build refuses to serve that store as the facility's past
— so the recorded store is dropped and the archiver pointed at the facility's
own appliance in the same step:

.. code-block:: bash

   osprey set connector=epics
   osprey set config.archiver.type=epics_archiver
   osprey set va_archiver=null

``osprey init`` does not offer the stand-in, and the build refuses
``control_system.type: live_standin`` on a profile that asks for no
``virtual_accelerator.live_standin``: both would name a machine the deployment
does not run.

**The label stays honest.** Nothing calls the stand-in the live machine. The
banner reads ``LIVE MACHINE (stand-in)``, the roster says the same, and the
header chip reads ``STAND-IN``. It carries a real machine's posture — the
same limits, the same approval prompts — because it is operated as one, not
because a write reaches the facility. What the stand-in rehearses is the procedure, not the risk.

**Telling the two machines apart.** Both run one image over one lattice, so the
stand-in ships a small fixed offset on its BPM readouts: a read that comes back
different is how you know which machine answered. That perturbation needs the
shipped built-in lattice behind it. A deployment whose environment pins
``VA_LATTICE=none``, or points the IOC at a facility channel file, serves that
manifest unperturbed instead — and the stand-in then reads identically to the
virtual accelerator beside it. The labels still tell them apart; the readings do
not.

**The archive belongs to the machine.** The recorder records the stand-in, and
the history seeded on the first deploy carries the same offsets the stand-in
reads — so its past and its present describe one machine, the way a real
machine's do. That is also why the live machine is gated while both are running:
see `Go live`_. ``osprey sim apply`` reaches both machines: a scenario changes
the world, not one lane.

On a laptop the second container is a real cost — the simulator image is
amd64-only, so on Apple Silicon a second one doubles what QEMU has to emulate,
for the life of the deployment. Delete the line when the rehearsal is over.

.. seealso::

   - :doc:`use-virtual-accelerator` — the simulator itself, the archive it
     deploys, and the one configuration the stack refuses.
   - :ref:`profile-virtual-accelerator` — the build-profile keys behind the
     simulator and the stand-in.
   - :doc:`/architecture/python-executor` — how executions are launched, and what the
     target stamp pins them to.
   - :doc:`../bluesky/index` — the plan stack that plan lanes belong to.
