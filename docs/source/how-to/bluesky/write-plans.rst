====================
Write Your Own Plans
====================

OSPREY ships three plans — an n-dimensional **grid scan**, an **orbit
response matrix** sweep, and a closed **orbit bump** sweep — and they are
deliberately generic. Your machine has its own measurements, and there are two
ways to add them: ask the agent to write one during a session, or install a
plan library that belongs to your facility.

The orbit bump is asked for in orbit space rather than in corrector currents:
name the three or four correctors allowed to act, the BPMs the beam should
move at and by how much, and the ones it must not move at all, and the plan
finds the kicks that do it — no lattice model needed. It walks the bump up and
back down step by step across the profile, verifying each step against the
tolerance you asked for — a tolerance narrower than the BPMs' own noise is
refused before anything moves.

Who is trusted, in one paragraph
================================

Plans are trusted by where they come from. Plans shipped with OSPREY, with a
preset, or installed by your facility run as they are. A plan the agent
writes mid-conversation — a **session plan** — is different: it runs only
after passing validation, and only as the *exact* version that passed. Change
one character and it must pass again. Nobody has to remember this rule; the
queue enforces it and refuses anything unvalidated, with a message that says
what to do.

Two ways to add a plan
======================

.. tab-set::

   .. tab-item:: Ask the agent

      Describe the measurement and let the agent do the authoring — it has a
      bundled skill (``writing-bluesky-plans``) for exactly this:

      .. code-block:: text

         Write me a plan that ramps one corrector while logging every
         BPM, and holds each setpoint for a settling time I can choose.

      The agent writes the plan file, runs it through the validator, and
      tells you the result. From there it is a normal plan: it appears in
      BLUESKY's Plans view, you review its parameters, and it queues and runs like
      any other — with its session-tier badge visible, so a reviewer always
      knows what they are looking at.

      Session plans are working drafts, not durable installations: they live
      with the running deployment, and after a restart of the bridge they
      must be validated again before they can run. A plan that earns its
      keep should graduate to your facility's library.

   .. tab-item:: Install a facility library

      Put your plan files in a directory and name it in your build profile:

      .. code-block:: yaml

         bluesky:
           plan_dir: plans/

      Every plan in it is installed read-only into the plan stack and
      trusted at **facility** tier — no per-session validation, available in
      every deployment built from the profile, listed in BLUESKY's Plans
      view and the agent's catalog like the shipped plans.

      To *remove* a plan from the catalog — shipped or otherwise — list it
      under ``excluded_plans`` in the same block, and it becomes invisible
      and non-runnable everywhere.

.. dropdown:: Anatomy of a plan file
   :color: info
   :icon: file-code

   A plan file is a small Python module with three parts:

   - **Metadata** — three things and no more: the plan's name, a human
     description, and whether it moves anything on the machine.
   - **Parameters** — a schema describing the knobs (names, types, limits).
     This is what BLUESKY's Plans view turns into a form, so a
     well-described parameter becomes a well-labeled field. Each parameter
     that holds channel names also says what the plan does with them — see
     *A plan says what it touches* below.
   - **The plan function** — builds the actual Bluesky plan from the
     parameters and the resolved devices.

   Plus an optional fourth: **the view** — a ``render`` function that turns
   the run's rows into the plan's own plots. See *Give a plan its own view*
   below.

   The agent's ``writing-bluesky-plans`` skill carries the full, current
   template — the fastest way to see one is to ask the agent to write a
   minimal plan and read the result.

.. dropdown:: What the validator checks
   :color: info
   :icon: check-circle

   Validation is static scrutiny first, then a rehearsal:

   1. **Static checks** — the file may only import what is on the
      validator's allowlist, and anything that reaches for the control
      system directly is rejected — all before a single line runs.
   2. **A dry run** — the plan is executed against mock devices in an
      isolated process, with all control-system access switched off. It has
      to run to completion there.

   A pass is recorded against the exact content of the file — its
   fingerprint — which is what makes the "exact bytes" rule enforceable.

.. dropdown:: Why an edited plan must pass again
   :color: info
   :icon: history

   The pass belongs to the fingerprint, not to the filename. Editing the file
   changes the fingerprint, so the old pass no longer applies — and the queue
   checks the fingerprint again both when a plan is added *and* when the
   queue starts, so there is no window where edited-but-unvalidated code can
   reach the machine. A bridge restart clears the recorded passes too, which
   is why a session plan that outlived a restart asks to be validated once
   more. Facility-tier plans carry no fingerprint bookkeeping — their trust
   comes from being installed by you.

Declare the devices a plan may drive
====================================

A plan names channels through its parameters, but *which* devices exist at all
is the deployment's answer, not the plan's. ``osprey build`` gives it from your
facility file: it writes ``data/bluesky_devices.yml`` into the build output, a
list of every device the queue server may drive or record, and the queue server
mounts that file unchanged.

.. code-block:: yaml

   # build/data/bluesky_devices.yml
   schema: osprey.facility.bluesky_devices/1
   settables:
     - name: SR:MAG:HCM:01:CURRENT:SP
       setpoint: SR:MAG:HCM:01:CURRENT:SP
       readback: SR:MAG:HCM:01:CURRENT:RB
   readables:
     - name: SR:MAG:HCM:01:CURRENT:RB
       pv: SR:MAG:HCM:01:CURRENT:RB
     - name: SR:DIAG:BPM:01:POSITION:X
       pv: SR:DIAG:BPM:01:POSITION:X

**Settables** are devices a plan may drive: one per setpoint channel of the
facility file, written to that channel and read back from the channel it is
paired with. A setpoint the facility pairs with no readback reads its own
setpoint. **Readables** are recorded and never written: one per readback
channel. A device's name is its address, which is also the column heading in
the run's data.

The build rewrites the file every time, so it is not a file to edit: to change
the device set, change the channels under ``data/facility/`` and build again.

When a plan drives a settable, the write is followed by a poll of the readback
until it reaches the demand. Two profile keys bound that wait:
``bluesky.settle_timeout_s`` (default 5.0 seconds) is how long the poll runs
before the move fails and the plan aborts, and ``bluesky.settle_tolerance``
(default ``1e-9``) is how close the readback must come, as an absolute
difference. The defaults suit a setpoint a controller echoes back exactly; a
device that physically moves — a magnet, an insertion-device gap — needs both
raised. Running out of budget always fails the plan: neither key can turn an
unsettled move into a successful one.

A deployment pointed at the ``mock`` control system drives no channels, so its
queue server comes up able to browse and describe plans and to run none of
them. That is a plain statement about the deployment, not a fault, and the
build says so in its output.

.. note::

   **The limits database is not the channel list.** ``channel_limits.json``
   states the range a write to a listed channel has to fall inside. It gates
   writes over a subset of the machine, and nothing *enumerates* the facility
   from it: a file listing a few hundred writable channels says nothing about
   the few thousand the facility has. The device set comes from the facility
   file, which lists every channel and says which way each one points.

.. note::

   **Two lanes, one device file.** A profile that turns on ``second_lane`` runs
   one plan lane for the live machine and one for the virtual accelerator — and
   both mount the *same* device file, because a facility has one namespace, not
   one per lane. A channel the file names that one lane does not serve fails
   that lane's pre-run probe, described below.

Deriving for a live lane is deliberate
--------------------------------------

The device set is the facility's namespace, and the live lane mounts it like
any other. That is a decision, not an oversight. A device in the worker's
namespace is a name a plan *may* reference; it is not a write that has
happened. The gates deciding whether a write lands sit on the write path — the
connector's per-write check and the bridge's arming and limits — and holding
the machine's own channels out of the namespace would add no gate to them. It
would only hide from the agent the channels it is allowed to *read*, and push
you back to hand-written device files that nothing keeps in step with the
facility.

One more thing stands behind that decision, and it is worth knowing:

**A run is refused before it moves anything.** A device file names channels; it
cannot promise the IOC serving them is up, and an unreachable channel used to
surface mid-plan — as a connection error out of the first read, with setpoints
already applied. The worker asks that question one message earlier instead:
before the plan is constructed, it probes every address the run's declared
devices touch — each setpoint, each readback, each recorded channel — on that
lane's own connector, and refuses the run if any of them does not answer.

.. code-block:: text

   refusing plan 'bump' before it moves anything — lane bluesky_live (target live):
   2 declared channels did not respond within 5 s: SR:MAG:HCM:07:CURRENT:SP, SR:MAG:HCM:07:CURRENT:RB

Nothing has been written when that arrives, so nothing is half-done. Each
address gets five seconds and one retry, and a channel that misses the first
probe and answers the second runs. The sweep as a whole is bounded too — 20
seconds for a small plan, up to 90 for one naming thousands of addresses — and
whatever it did not reach inside that bound is reported separately from what it
asked and got no answer from. Both refuse the run; they are different findings
and the message keeps them apart. It is asked per *run* rather than at enqueue
time: what a queued plan needs is channels that are alive when it runs, and the
gap between the two can be an IOC restart.

Listed as settable, refused at write time
-----------------------------------------

Being a settable says the facility describes the channel as one that is
written. It does not say a write to it will be accepted. Those are two
questions, answered in two places, and a facility whose two answers differ is a
normal deployment rather than a misconfiguration:

- The **facility's description** — the graph corpus, or the channel-finder
  database — answers membership: which channels exist. In graph mode it answers
  direction too, from the binding's own ``writesSignal``/``readsSignal``.
- The **limits database** answers permission, at the moment a write is
  attempted: what range the value has to be in, and whether a channel it does
  not list may be written at all (``control_system.limits_checking.mode``).

So a graph corpus may mark a channel settable that the limits database refuses
— a facility's structural description and its enforced write ranges are
maintained separately, and OSPREY does not reconcile them at build time. The
plan may name that device, and the write is refused when it is attempted, by
the target that refused it, with a message saying so. The build does not
intersect the two lists, and it does not drop a channel from the namespace
because the limits file omits it — that would report a smaller machine than the
facility has. (On a database paradigm the same limits file is what supplied the
direction in the first place, so the two agree on which channels are settable
by construction; what they can still disagree about is the value.)

Which channels count as scan devices is no longer a separate answer from which
channels your facility has: the queue server's device set and the channel
finder read the same source, the one the paradigm in force selects. In graph
mode that source *is* the corpus, so the device set, the channel finder and the
knowledge graph are three views of one description and cannot drift apart. On a
database paradigm the graph is generated from the same channel database
(:doc:`../facility-knowledge/use-facility-graph`), so the three agree as long
as the corpus is regenerated when the database changes.

A plan says what it touches
===========================

A plan's parameters name channels, but a list of names on its own does not say
whether the plan will *drive* those channels or only *record* them. Every plan
file answers that outright: a parameter holding channel names is marked either
**movable** — the plan drives it to a value — or **readable** — the plan
records it without changing it.

That one marking is what the rest of OSPREY works from. It decides which
stand-in devices the validator builds for the rehearsal, which names are
checked against your machine before a plan is queued, what the approval prompt
shows the human who is about to say yes, and which channel the default plot
uses for its x axis. Each of those used to guess from how a parameter was
spelled. Now the plan says it once, and everything reads the same answer.

.. raw:: html
   :file: ../../_diagrams/plan-parameter-marking.html

Two consequences you will notice:

- **The names are yours.** Call the parameters whatever your facility calls
  them — correctors, BPMs, setpoints, monitors. The marking carries the
  meaning, so nothing downstream depends on the spelling.
- **A plan that moves the machine has to show what it moves.** A plan whose
  metadata says it writes, but which marks nothing as movable, is refused when
  the catalog loads it and never appears — with a message saying exactly that.
  Such a plan must also open a run and state how many points that run will
  take; that number is what live progress counts against. A plan built on top
  of one of Bluesky's own scans inherits the run and its point count from that
  scan, so it states neither itself — but it still marks its own parameters,
  because those markings are what everything else reads.

Give a plan its own view
========================

Every run gets a figure in the BLUESKY panel, and by default it is drawn for
you: every numeric column the run recorded, plotted against the channel the
plan drives — or simply in the order the readings were taken, when a plan
drives more than one. That **default view** is honest and, for a
straightforward measurement, enough.

A plan that measures something the raw columns cannot show can bring its own
view instead — a small ``render`` function that receives the run's rows and its
parameters and returns the plots the plan itself designs. The shipped ``orm``
plan does exactly that: a trace per corrector while the sweep runs, then the
fitted response matrix and per-device scores once there is enough data. So does
``orbit_bump_sweep``: the orbit shift across the BPMs at each amplitude step,
the residual against its tolerance band, and where the correctors sat while it
walked — plus the monitors' response, on a run that was given extra monitor
channels at all. A panel with nothing to draw is left out rather than drawn
empty.

The vocabulary is small on purpose. A figure is a list of **panels**; each
panel has a title, axis labels and units, any notes worth printing beside it,
and exactly one **mark**:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Mark
     - What it draws
   * - **Lines**
     - Named series of x/y points — a sweep, a trend, one line per monitor. A
       reading the run never took stays a gap in the line, never a zero.
   * - **Bars**
     - One value per named category — a score or a total per device.
   * - **Heatmap**
     - A labelled 2-D grid — for example BPMs against correctors, each cell a
       fitted slope.

Three rules keep a view honest, and the framework enforces all three:

- **Drawing never disturbs a plan.** A view is computed from data already
  recorded, after the fact. If it fails, the run and its numbers are untouched
  and the panel simply shows the default view with a note saying why.
- **Views name no facility.** Labels come from the plan's parameters and the
  columns the run recorded, so the same plan draws correct device names at any
  facility that installs it.
- **Only installed plans draw their own view.** A plan's ``render`` runs inside
  the bridge every time a panel refreshes, so it is honored for plans shipped
  with OSPREY, with a preset, or installed by your facility — not for session
  plans the agent writes mid-conversation. A session plan queues, runs and
  records data exactly as any other; its runs just show the default view. A
  view is one more reason for a plan that earns its keep to graduate into your
  facility's library.

.. note::

   **Views apply going forward.** A figure is computed by the plan code that
   owns the plan's name *now*, so adding a ``render`` — or fixing one — shows
   up on the next run with nothing to migrate. The exception is old data: a run
   recorded before OSPREY kept track of which plan produced it has nothing to
   tie it back to plan code, so it keeps showing the default view whatever you
   add later. Its numbers are all still there; only the plan's own view is out
   of reach.

.. seealso::

   :doc:`queue`
      How a queued plan actually runs, and what refusals mean.

   :doc:`/how-to/build-profiles`
      The build profile that owns ``plan_dir`` and ``excluded_plans``.
