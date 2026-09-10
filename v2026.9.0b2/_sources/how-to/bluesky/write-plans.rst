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
is the deployment's answer, not the plan's. One short file gives it: a list of
every device the queue server may drive or record, named in the build profile
by ``bluesky.devices_file`` — ``data/bluesky_devices.yml`` inside your project
unless you point it elsewhere.

.. code-block:: yaml

   # data/bluesky_devices.yml
   settables:
     - name: SR:MAG:HCM:01
       setpoint: SR:MAG:HCM:01:CURRENT:SP
       readback: SR:MAG:HCM:01:CURRENT:RB
     - name: BTS:QF3
       setpoint: BTS:QF3,PS:CURRENT:SP   # a comma in an address is fine, and
                                         # no readback means "read the setpoint"

   readables:
     - name: SR:DIAG:BPM:01:X
       pv: SR:DIAG:BPM:01:X:RB

**Settables** are devices a plan may drive: a name, the channel a value is
written to, and — optionally — a separate channel it is read back from. Leave
the readback out and the device reads its own setpoint. **Readables** are
recorded and never written. The name is what a plan and the agent refer to, and
it is also the column heading in the run's data, so each name may appear only
once across both lists.

Nothing in the file is split on any character, which is the point of writing it
this way: an address containing a comma — as some real magnet power supplies
have — is written out plainly and needs no escaping.

``osprey build`` reads the file and refuses to build on an entry it cannot use,
naming ``bluesky.devices_file`` and the entry at fault. That refusal is
deliberate: the queue server itself skips a bad entry with a warning, so a
deployment built from an unchecked file would come up looking healthy while
missing exactly the devices you meant to add.

A deployment ends up with its device set in one of three ways:

- **You write the file.** The normal case at a real facility. A path inside the
  project travels with the built deployment; an absolute path is yours alone —
  OSPREY reads it where it is and never rewrites or moves it. Either way the
  build takes its own copy for the queue server to mount, so an edit to your
  file reaches a running deployment at the next ``osprey build``.
- **The build derives one from your facility's own description.** A deployment
  on a real control system, with no file yet at a project path, gets a device
  set derived from the same description of the machine the channel finder
  reads: the knowledge graph corpus in graph mode, and otherwise the
  channel-finder database for the paradigm in force —
  ``channel_finder.pipeline_mode`` where it is set, and otherwise whichever
  pipeline has a database configured. That source says which channels exist.
  Which of them are *settable* depends on which source it is, and the two
  differ:

  - **A graph corpus says so itself.** Every binding carries a direction —
    ``writesSignal`` or ``readsSignal`` — so the split into settables and
    readables is read out of the corpus rather than inferred.
  - **A channel-finder database does not.** No paradigm database records a
    direction, so the build derives one: from the write-limits file
    (``control_system.limits_checking.database_path``) when the deployment has
    one, read exactly as the write path reads it; failing that, from the
    address grammar, where a final ``:SP`` token is a setpoint and everything
    else is read. With neither — no limits file, and no ``:SP`` address in the
    database — the build stages nothing and says the channels are known but
    their directions are not.

  A settable's readback is address grammar in both cases: the final ``:SP``
  becomes ``:RB``, and that readback is adopted only if the source enumerates
  that address as a read channel. Where it does not, the device carries no
  readback and the worker reads the setpoint back. The build says which source
  it read and how many devices came out of it — and, where some channels'
  directions could not be stated, how many it left out, so a smaller device set
  is never a silently smaller machine. The derived set is written straight into
  the build output and rewritten every time you build, so it is not a file to
  edit — to take over, put your own file at the path the key names. An
  **absolute** ``devices_file`` with no file at it derives nothing: that path
  says an operator supplies the file, so its absence means "not staged yet",
  not "choose a device set for me".
- **Neither.** The queue server comes up able to browse and describe plans and
  to run none of them. That is a plain statement about the deployment, not a
  fault, and the build says so in its output, in the words of whatever was
  missing: no channel source configured at all, no ``bluesky.devices_file``
  configured, graph mode naming no readable corpus, a source that is not there,
  one that was read and declares no channel, or one whose channels are known
  while which of them are settable is not. A deployment pointed at the ``mock``
  control system is browse-only this way whatever file is present.

One case is neither derived nor browse-only: a roster source that is **there
and unreadable** refuses the build. An absent source is a facility this project
did not describe; a corrupt one is a facility it meant to describe and got
wrong, and deriving past it would stage a partial device set that looks exactly
like a complete one.

.. note::

   **The limits database is not the channel list.** ``channel_limits.json``
   states the range a write to a listed channel has to fall inside. It gates
   writes over a subset of the machine, and nothing *enumerates* the facility
   from it: a file listing a few hundred writable channels says nothing about
   the few thousand the facility has. It does still answer one question about a
   derived set — outside graph mode it is what says which of the enumerated
   channels are settable, as above. That is the practical argument for a graph
   corpus where you have one: it carries the direction itself, so the device
   set does not inherit the shape of a limits file.

.. note::

   **Two lanes, one device file.** A profile that turns on ``second_lane`` runs
   one plan lane for the live machine and one for the virtual accelerator — and
   both mount the *same* device file, because a facility has one namespace, not
   one per lane. So a lane can only drive the devices that file names: a file
   written for the live machine leaves the virtual-accelerator lane with
   nothing to address unless those channels are served there too.

Deriving for a live lane is deliberate
--------------------------------------

The derived set is the facility's namespace, and the live lane mounts it like
any other. That is a decision, not an oversight. A device in the worker's
namespace is a name a plan *may* reference; it is not a write that has
happened. The gates deciding whether a write lands sit on the write path — the
connector's per-write check and the bridge's arming and limits — and holding
the machine's own channels out of the namespace would add no gate to them. It
would only hide from the agent the channels it is allowed to *read*, and push
you back to hand-written device files that nothing keeps in step with the
facility.

Two things stand behind that decision, and both are worth knowing:

**The build refuses an armed lane with no limits.** When a lane's device set is
derived, and a *deployed* lane's control target has writes armed while limits
checking is not switched on for the same target, ``osprey build`` stops and
names both keys. A lane the profile does not deploy is not examined. Limits checking that is off builds no validator at all, so every derived
device would take whatever value a plan asked for with no database consulted.
Authored device files are exempt — that set is your own list. The refusal is
per lane and per target: a virtual-accelerator lane with limits on does not
excuse a live lane without them.

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
  not list may be written at all
  (``control_system.limits_checking.allow_unlisted_channels``).

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
      The build profile that owns ``plan_dir``, ``excluded_plans`` and
      ``devices_file``.
