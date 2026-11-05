====================
Write Your Own Plans
====================

This page shows you two ways to add your own measurement plans: ask the agent
to write one during a session, or install a plan library that belongs to your
facility.

OSPREY ships three deliberately generic plans:

- an n-dimensional **grid scan**;
- an **orbit response matrix** sweep;
- a closed **orbit bump** sweep.

You ask for the orbit bump in orbit space rather than in corrector currents.
Name the three or four correctors allowed to act, the BPMs where the beam
should move and by how much, and the BPMs where it must not move at all. The
plan finds the kicks that do it, with no lattice model needed. It walks the
bump up and back down step by step across the profile, and verifies each step
against the tolerance you asked for. A tolerance narrower than the BPMs' own
noise is refused before anything moves.

Who is trusted, in one paragraph
================================

Plans are trusted by where they come from:

- Plans shipped with OSPREY, with a preset, or installed by your facility run
  as they are.
- A plan the agent writes mid-conversation is a **session plan**. It runs only
  after passing validation, and only as the *exact* version that passed.
  Change one character and it must pass again.

You don't have to remember this rule. The queue enforces it and refuses
anything unvalidated, with a message that says what to do.

Two ways to add a plan
======================

.. tab-set::

   .. tab-item:: Ask the agent

      Describe the measurement and let the agent write the plan. It has a
      bundled skill (``writing-bluesky-plans``) for exactly this:

      .. code-block:: text

         Write me a plan that ramps one corrector while logging every
         BPM, and holds each setpoint for a settling time I can choose.

      The agent writes the plan file, runs it through the validator, and
      tells you the result. From there it is a normal plan. It appears in
      BLUESKY's Plans view, you review its parameters, and it queues and runs
      like any other. Its session-tier badge stays visible, so a reviewer
      always knows what they are looking at.

      Session plans are working drafts, not durable installations. They live
      with the running deployment. After a restart of the bridge, they must
      be validated again before they can run. Move a plan you want to keep
      into your facility's library.

   .. tab-item:: Install a facility library

      #. Put your plan files in a directory, for example ``plans/``.

      #. Name the directory in your build profile:

         .. code-block:: yaml

            bluesky:
              plan_dir: plans/

      #. Run ``osprey build``. Every plan in the directory is installed
         read-only into the plan stack.

      #. Open BLUESKY's Plans view and check that your plans are listed next
         to the shipped plans.

      Facility plans are trusted at **facility** tier. They need no
      per-session validation, and they are available in every deployment
      built from the profile. They are listed in BLUESKY's Plans view and the
      agent's catalog like the shipped plans.

      To *remove* a plan from the catalog, shipped or otherwise, list it
      under ``excluded_plans`` in the same block. The plan becomes invisible
      and non-runnable everywhere.

.. dropdown:: Anatomy of a plan file
   :color: info
   :icon: file-code

   A plan file is a small Python module with three parts:

   - **Metadata**: three things and no more. These are the plan's name, a
     human description, and whether it moves anything on the machine.
   - **Parameters**: a schema describing the knobs (names, types, limits).
     BLUESKY's Plans view turns this schema into a form, so a well-described
     parameter becomes a well-labeled field. Each parameter that holds
     channel names also says what the plan does with them. See *A plan says
     what it touches* below.
   - **The plan function**: builds the actual Bluesky plan from the
     parameters and the resolved devices.

   A plan file can also have an optional fourth part, **the view**. The view
   is a ``render`` function that turns the run's rows into the plan's own
   plots. See *Give a plan its own view* below.

   The agent's ``writing-bluesky-plans`` skill carries the full, current
   template. To see one, ask the agent to write a minimal plan and read the
   result.

.. dropdown:: What the validator checks
   :color: info
   :icon: check-circle

   Validation is static scrutiny first, then a rehearsal:

   1. **Static checks**: the file may only import what is on the
      validator's allowlist. Anything that reaches for the control system
      directly is rejected. These checks finish before a single line runs.
   2. **A dry run**: the plan is executed against mock devices in an
      isolated process, with all control-system access switched off. It has
      to run to completion there.

   A pass is recorded against the exact content of the file, its
   fingerprint. The fingerprint is what makes the "exact bytes" rule
   enforceable.

.. dropdown:: Why an edited plan must pass again
   :color: info
   :icon: history

   The pass belongs to the fingerprint, not to the filename. Editing the file
   changes the fingerprint, so the old pass no longer applies. The queue
   checks the fingerprint both when a plan is added *and* when the queue
   starts. There is no window where edited but unvalidated code can reach the
   machine.

   A bridge restart clears the recorded passes too. That is why a session
   plan that outlived a restart asks to be validated once more. Facility-tier
   plans carry no fingerprint bookkeeping. Their trust comes from being
   installed by you.

Declare the devices a plan may drive
====================================

A plan names channels through its parameters, but the deployment decides
*which* devices exist at all. ``osprey build`` derives the device set from your
facility file. It writes ``data/bluesky_devices.yml`` into the build output: a
list of every device the queue server may drive or record. The queue server
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

The file has two lists:

- **Settables** are devices a plan may drive. There is one per setpoint
  channel of the facility file. A settable is written to that channel and read
  back from the channel it is paired with. A setpoint the facility pairs with
  no readback reads its own setpoint.
- **Readables** are recorded and never written. There is one per readback
  channel.

A device's name is its address, which is also the column heading in the run's
data.

Don't edit this file: the build rewrites it every time. To change the device
set, change the channels under ``data/facility/`` and build again.

A plan the agent writes during a session names channels from this same set.
``list_devices`` reads it. The pre-enqueue check refuses a plan that names a
device the file does not declare. Validation alone cannot catch this, because
it builds a mock for whatever the sample arguments name, so it says nothing
about what the worker has. The channel finder answers for the whole facility.
This file answers for the queue worker, and a plan's channels come from here.

When a plan drives a settable, the move waits for its readback to come within
a bound of the demand. The setpoint's ``tolerance`` in the facility file, next
to its ``pair``, sets the bound:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Setpoint's ``tolerance``
     - The readback must come within
   * - ``tolerance: {absolute: <x>}``
     - ``<x>`` of the demand, in the channel's unit.
   * - ``tolerance: {relative: <f>}``
     - ``<f>`` times the demand (a fraction of the demand).
   * - None, on a lane that serves the simulator
     - Its readback's motion band: ``|drift.amplitude|`` + 6 × the noise
       sigma. A ``relative`` sigma is taken at the seed's ``nominal``.
   * - None, on a live or stand-in lane
     - The floor.

The floor is ``bluesky.settle_tolerance`` (default ``1e-9``), an absolute
difference, and the bound is never tighter. If the readback is not inside its
bound after ``bluesky.settle_timeout_s`` (default 5.0 seconds), the move fails
and the plan aborts. pyAML measurement steps wait for their readbacks by the
same rule, without the floor.

The build warns:

- once per live or stand-in lane, with the count of setpoints that declare no
  ``tolerance``;
- once per scenario that moves a readback beyond its setpoint's tolerance,
  since that scenario's moves time out while it is active.

``osprey facility validate`` prints the same warnings.

A device that physically moves needs a ``tolerance`` on its setpoint, and a
slow device needs ``settle_timeout_s`` raised. Running out of budget always
fails the plan: neither key can turn an unsettled move into a successful one.

A deployment serving the simulator in process drives no channels over Channel
Access. Its queue server comes up able to browse and describe plans, and to
run none of them. That is a plain statement about the deployment, not a
fault, and the build says so in its output.

.. note::

   **The limits database is not the channel list.** The limits are the
   records of ``data/facility/limits.yaml``. ``osprey build`` renders them
   into ``build/data/channel_limits.json``. The connector's reference monitor
   and the runtime's limits check enforce that file on every write.

   The limits database gates writes over a subset of the machine, and nothing
   *enumerates* the facility from it. A file listing a few hundred writable
   channels says nothing about the few thousand the facility has. The device
   set comes from the facility file, which lists every channel and says which
   way each one points.

.. note::

   **Two lanes, one device file.** A profile that turns on ``second_lane`` runs
   one plan lane for the live machine and one for the virtual accelerator.
   Both mount the *same* device file, because a facility has one namespace,
   not one per lane. If the file names a channel that one lane does not serve,
   that channel fails the lane's pre-run probe, described below.

Deriving for a live lane is deliberate
--------------------------------------

The device set is the facility's namespace, and the live lane mounts it like
any other. That is a decision, not an oversight. A device in the worker's
namespace is a name a plan *may* reference. It is not a write that has
happened.

The gates that decide whether a write lands sit on the write path: the
connector's per-write check, and the bridge's arming and limits. Holding the
machine's own channels out of the namespace would add no gate to them. It
would only hide from the agent the channels it is allowed to *read*. It would
also push you back to hand-written device files that nothing keeps in step
with the facility.

One more thing stands behind that decision:

**A run is refused before it moves anything.** A device file names channels,
but it cannot promise that the IOC serving them is up. An unreachable channel
used to surface mid-plan, as a connection error out of the first read, with
setpoints already applied. The worker now asks that question one message
earlier. Before the plan is constructed, the worker probes every address the
run's declared devices touch: each setpoint, each readback, each recorded
channel. It probes on that lane's own connector, and refuses the run if any
address does not answer.

.. code-block:: text

   refusing plan 'bump' before it moves anything — lane bluesky_live (target live):
   2 declared channels did not respond within 5 s: SR:MAG:HCM:07:CURRENT:SP, SR:MAG:HCM:07:CURRENT:RB

Nothing has been written when that message arrives, so nothing is half-done.
Each address gets five seconds and one retry. A channel that misses the first
probe and answers the second runs.

The sweep as a whole is bounded too: 20 seconds for a small plan, up to 90 for
one naming thousands of addresses. The message reports addresses the sweep did
not reach inside that bound separately from addresses it asked and got no
answer from. Both refuse the run, but they are different findings.

The probe runs per *run* rather than at enqueue time. A queued plan needs
channels that are alive when it runs, and the gap between enqueue and run can
be an IOC restart.

Listed as settable, refused at write time
-----------------------------------------

Being a settable says the facility describes the channel as one that is
written. It does not say a write to it will be accepted. Two places answer
those two questions. A facility whose two answers differ is a normal
deployment, not a misconfiguration:

- The **facility's description** (the graph corpus, or the channel-finder
  database) answers membership: which channels exist. In graph mode it
  answers direction too, from the binding's own
  ``writesSignal``/``readsSignal``.
- The **limits database** answers permission, at the moment a write is
  attempted. It says what range the value has to be in, and whether a channel
  it does not list may be written at all
  (``control_system.limits_checking.mode``).

So a graph corpus may mark a channel settable that the limits database
refuses. A facility maintains its structural description and its enforced
write ranges separately, and OSPREY does not reconcile them at build time. The
plan may name that device. When the write is attempted, the target refuses it
with a message saying so.

The build does not intersect the two lists. It does not drop a channel from
the namespace because the limits file omits it, since that would report a
smaller machine than the facility has. On a database paradigm, the same limits
file supplied the direction in the first place. There the two agree on which
channels are settable by construction, and can still disagree about the
value.

The queue server's device set and the channel finder read the same source:
the one the paradigm in force selects. Which channels count as scan devices is
therefore the same answer as which channels your facility has.

- In graph mode, that source *is* the corpus. The device set, the channel
  finder and the knowledge graph are three views of one description, and they
  cannot drift apart.
- On a database paradigm, the graph is generated from the same channel
  database (:doc:`../facility-knowledge/use-facility-graph`). The three agree
  as long as the corpus is regenerated when the database changes.

A plan says what it touches
===========================

A plan's parameters name channels, but a list of names on its own does not say
whether the plan *drives* those channels or only *records* them. Every plan
file answers that outright. A parameter holding channel names is marked as
one of:

- **movable**: the plan drives it to a value;
- **readable**: the plan records it without changing it.

The rest of OSPREY works from that one marking. It decides:

- which stand-in devices the validator builds for the rehearsal (the dry run);
- which names are checked against your machine before a plan is queued;
- what the approval prompt shows the person who is about to say yes;
- which channel the default plot uses for its x axis.

Each of those used to guess from how a parameter was spelled. Now the plan
says it once, and everything reads the same answer.

.. raw:: html
   :file: ../../_diagrams/plan-parameter-marking.html

Two consequences you will notice:

- **The names are yours.** Call the parameters whatever your facility calls
  them: correctors, BPMs, setpoints, monitors. The marking carries the
  meaning, so nothing downstream depends on the spelling.
- **A plan that moves the machine has to show what it moves.** If a plan's
  metadata says it writes but the plan marks nothing as movable, the catalog
  refuses the plan when it loads it. The refusal message says exactly that,
  and the plan never appears.

  Such a plan must also open a run and state how many points that run will
  take. Live progress counts against that number. A plan built on top of one
  of Bluesky's own scans inherits the run and its point count from that scan,
  so it states neither itself. It still marks its own parameters, because
  everything else reads those markings.
- **A plan records what it moves.** Every reading is taken at some setting of
  the channel being moved, and only means something next to it. The validator
  watches the rehearsal. It refuses a plan that moves a channel without ever
  recording it alongside its readings, because such a run would finish with
  no record of what each reading was taken at.

Give a plan its own view
========================

Every run gets a figure in the BLUESKY panel, and by default it is drawn for
you. The **default view** plots every numeric column the run recorded against
the channel the plan drives. When a plan drives more than one channel, the
default view plots the columns in the order the readings were taken. For a straightforward
measurement, the default view is honest and enough.

A plan that measures something the raw columns cannot show can bring its own
view instead. The view is a small ``render`` function. It receives the run's
rows and its parameters, and returns the plots the plan itself designs. Two
shipped plans do this:

- ``orm`` draws a trace per corrector while the sweep runs. Once there is
  enough data, it draws the fitted response matrix and per-device scores.
- ``orbit_bump_sweep`` draws the orbit shift across the BPMs at each
  amplitude step, the residual against its tolerance band, and where the
  correctors sat while it walked. On a run that was given extra monitor
  channels, it also draws the monitors' response.

A panel with nothing to draw is left out rather than drawn empty.

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
  recorded, after the fact. If it fails, the run and its numbers are untouched.
  The panel shows the default view with a note saying why.
- **Views name no facility.** Labels come from the plan's parameters and the
  columns the run recorded, so the same plan draws correct device names at any
  facility that installs it.
- **Only installed plans draw their own view.** A plan's ``render`` runs inside
  the bridge every time a panel refreshes. So it is honored for plans shipped
  with OSPREY, with a preset, or installed by your facility. It is not honored
  for session plans the agent writes mid-conversation. A session plan queues,
  runs and records data exactly as any other, but its runs show the default
  view. A view is one more reason to move a plan you want to keep into your
  facility's library.

.. note::

   **Views apply going forward.** The plan code that owns the plan's name
   *now* computes the figure. Adding or fixing a ``render`` shows up on the
   next run, with nothing to migrate.

   The exception is old data. A run recorded before OSPREY kept track of which
   plan produced it has nothing to tie it back to plan code. It keeps showing
   the default view, whatever you add later. Its numbers are all still there;
   only the plan's own view is out of reach.

.. seealso::

   :doc:`queue`
      How a queued plan actually runs, and what refusals mean.

   :doc:`/how-to/build-profiles`
      The build profile that owns ``plan_dir`` and ``excluded_plans``.
