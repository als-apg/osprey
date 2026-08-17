=========================
Write Your Own Scan Plans
=========================

OSPREY ships two scan plans — an n-dimensional **grid scan** and an **orbit
response matrix** sweep — and they are deliberately generic. Your machine has
its own measurements, and there are two ways to add them: ask the agent to
write one during a session, or install a plan library that belongs to your
facility.

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

         Write me a scan plan that ramps one corrector while logging every
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

      Every plan in it is installed read-only into the scan stack and
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

A plan says what it touches
===========================

A plan's parameters name channels, but a list of names on its own does not say
whether the plan will *drive* those channels or only *record* them. Every plan
file answers that outright: a parameter holding channel names is marked either
**movable** — the plan drives it to a value — or **readable** — the plan
records it without changing it.

That one marking is what the rest of OSPREY works from. It decides which
stand-in devices the validator builds for the rehearsal, which names are
checked against your machine before a scan is queued, what the approval prompt
shows the human who is about to say yes, and which channel the default plot
uses for its x axis. Each of those used to guess from how a parameter was
spelled. Now the plan says it once, and everything reads the same answer.

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
fitted response matrix and per-device scores once there is enough data.

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

- **Drawing never disturbs a scan.** A view is computed from data already
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
