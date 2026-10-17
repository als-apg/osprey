.. _thread-the-first-turn:

=====================
Thread the First Turn
=====================

First-turn threading gets an injected beam around its first turn. You take a
**shot** --- arm the BPMs, fire one injection, read where the beam went ---
and the trajectory service computes a corrector **step** from it. You
**apply** the step, take the next shot, and repeat until no BPM loses the
beam. This page says what a deployment needs for that loop, how the agent and
the operator drive it, and what the machine is left with when an action stops
part-way. The exact routes and refusal codes are in
:ref:`reference-trajectory-service`.

What the loop needs
===================

- **A trajectory service.** A profile with a ``trajectory:`` block makes
  ``osprey up`` deploy it (see :ref:`architecture-trajectory-service`). It
  claims one control target, the deployment's baseline unless the profile
  names another.
- **A trajectory response matrix.** The service solves its step with pyAML's
  ``Trajectory`` tool over the first-turn response matrix the build writes
  for the model. A model whose measurement file lists ``trajectory`` gets one
  (see :doc:`describe-your-facility`). Without it the service starts, and
  every shot and apply is refused ``tool_unavailable``.
- **A sequence module.** The facility's code that fires an injection, named
  by ``services.trajectory.sequence_dir``. A virtual-accelerator target gets
  the shipped one; any other target names its own, and until it does every
  shot is refused ``no_sequence_module``. What the module holds is in
  :ref:`reference-trajectory-sequence`.
- **Writes armed on the target**, and an owner whose control context names
  that target and is not narrowed. The service checks both for every writing
  action, as it reaches the service's worker.
- **The trajectory tab, for the agent.** The agent's ``trajectory`` server is
  rendered only for a persona that lists ``- trajectory`` in its web panels;
  the ``control-assistant-readwrite`` and ``control-assistant-admin`` presets
  do. Such a persona must deny both ``Bash`` and ``Monitor``: a deploy that
  would show the tab to an agent with a shell is refused, with no waiver,
  because a shell could read the agent's panel token and call the service
  without an approval prompt.

The loop
========

Shot
----

A shot takes ``n_shots`` injections (one by default) inside one action, under
the target's guarded-run lock. Each injection runs in this order: read the
BPM records' timestamps, arm every BPM in one command batch, wait until every
BPM reports armed, run the sequence module's ``inject``, wait until every BPM
reports done, read the turn-by-turn records, and extract the first turn. A
record not stamped after the arm, or a BPM that never armed, fails that
injection: it is listed with its reason, left out, and never retried.

The included injections are averaged, ``mean`` or ``nanmean``. The shot
answers each BPM's ``x``, ``y`` and ``sum``, the BPMs where the beam was
**lost**, the last BPM it reached, the injections included and failed, and
the commands fired. The first BPM whose sum does not rise by
``services.trajectory.loss_sum_cutoff``, and every BPM after it, are lost; an
injection whose first BPM's sum reads less than
``services.trajectory.shot_sum_threshold`` failed. Both are in the units of
the facility's BPM sum record (see :doc:`/reference/configuration/config`).

A shot is refused before it starts when its worst case does not fit
``services.trajectory.action_timeout_s``, so ask for fewer injections rather
than a longer timeout. A shot moves no setting: everything it writes is a
command.

Step
----

Each shot that included at least one injection leaves one **pending step**:
the correction the tool solves from the shot against the current reference
trajectory (zero when no reference is selected). The step is the increment
``δ`` written to each corrector, scaled down uniformly by a ``factor`` of at
most 1 so that no ``|δ|`` reaches its corrector's ``max_step``. It also names
the correctors the tool zeroed or could not solve.

The step carries a **digest** that binds the target, the model, the session,
the shot, the reference and every ``δ``. An apply names the step by that
digest, so it writes exactly the step that was read. A new shot or a new
reference replaces the pending step; a take-over or a put-back voids it.

Apply
-----

An apply writes the pending step: each corrector with ``δ ≠ 0`` is set to the
value read just before plus ``δ``. It is refused, with nothing written, when
the digest is not the pending step's, the step's shot is not the latest, the
step was already applied, every ``δ`` is zero, any new value breaks the
target's limits, or a Bluesky plan runs on the target.

The settings are then written one at a time. The answer's ``moved`` lists
every write that was not refused, each with its value ``before`` and its
outcome --- also a write that failed, since it may have landed. Then take the
next shot.

Put-back
--------

A put-back sets named settings back to values you give it, typically the
``before`` values an earlier apply listed in ``moved``. It names the service's
model and a list of ``{"name", "before"}`` entries; each name is a corrector
of the response matrix or an address of a slot the sequence module declares.
The service keeps no record of earlier values: a put-back's values arrive in
the call, so keep the ``moved`` list an action answered. A put-back voids the
pending step and is held to the same limits as an apply.

Driving it from the agent
=========================

The agent drives the loop with three tools on its ``trajectory`` server (see
:ref:`trajectory-mcp-server`):

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Tool
     - Approval
     - What it does
   * - ``trajectory_shot``
     - Asked
     - Takes a shot and leaves the pending step.
   * - ``trajectory_step``
     - Never asked
     - Reads the pending step and its digest. It moves nothing.
   * - ``trajectory_apply``
     - Asked
     - Writes the pending step by its digest, or puts values back.

A shot's prompt shows the service's own target and the shots asked for. An
apply's prompt shows the target and the pending step as the service holds it
--- its digest, the solve's reason, the factor, every corrector ``δ`` and the
correctors zeroed or unsolved --- fetched from the service while the prompt
is drawn. A put-back's prompt shows every name and the value it goes back to.

What you approve is what runs. The prompt binds the call to the target it
showed and, for an apply, to the step's digest or to a digest of the put-back
values; the server refuses the call ``approval_missing``, writing nothing,
when the service now drives another target or the call carries another
digest. A prompt that cannot show what the call would act on is not asked at
all: the call is denied, and the agent is told to read the step again. These
tools are never offered to a dispatched job.

The agent opens its trajectory session on its first shot or apply, keeps it
while its server runs, and releases it when the server exits. A session its
own earlier server left behind is taken over; one anybody else holds is
refused with the holder named.

Stopping an action
==================

A **Stop** ends the action in flight. Any owner the service admits may send
one, through either door, without holding the session; the holder keeps the
session. It is answered at once, ``202`` with the stop budget
(``stop_budget_s``), and the action ends at its next checkpoint within that
budget: the sequence module's longest pause, one connector call and the
checkpoint poll.

Stop is cooperative. It is checked between writes and at every call the
sequence module makes; nothing cuts a write off half-way, and a write already
sent is waited for. What a stopped action leaves behind:

- **A shot** fired the commands it lists in ``commands_fired``, writes no shot
  file and leaves no step.
- **An apply or a put-back** wrote the settings its ``moved`` lists, each with
  its ``before``, and stopped. The service does not restore them; put them
  back with a put-back of that list if you want them back.
- **A setup** lists what it moved the same way.

An apply or a put-back that runs past ``action_timeout_s``, or whose write
fails or is refused, ends the same way, listing what it moved.

A **take-over** stops the old holder's action the same way and voids its
pending step. It is refused ``apply_in_progress`` while an apply or a
put-back is writing.

The operator's panel door
=========================

The operator reaches the service through the trajectory tab, the **panel
door**. It serves everything the agent's tools reach, and these besides:

- **Setup** runs the sequence module's ``bpm_mode`` and, when asked,
  ``bunch_clean``, under the run lock and outside any shot. It never arms and
  never fires an injection.
- **A continuous run** takes shots one after another, each its own action:
  the run lock is released between shots, so other queued actions run in
  between, and the step is recomputed after every shot. It never applies a
  step. It holds a lease that the panel's heartbeats renew; one that lapses
  ends the run. It also ends on a Stop or take-over, after
  ``services.trajectory.continuous_max_shots`` shots, when the latest step is
  all zero, or when a shot raises an error. While it runs, setup and every apply are
  refused ``continuous_running``.
- **References**: save a shot as a named reference trajectory, list them,
  load one and select the one the step is solved against.
- **The shot listing** and **the state stream**, which pushes a fresh state
  after every change.

The agent's panel token cannot reach the panel door: it is the operator's own
browser.

Who acts, and what is recorded
==============================

Every action is the terminal owner's, whichever door it took: the service
names the account the terminal's secret maps to and ignores any name a
request claims. Every writing action --- open, take-over, release, Stop,
shot, setup, apply, put-back, continuous start, reference save and select ---
is one record in ``trajectory_service.jsonl`` under ``var/audit/trajectory/``,
with the owner as ``actor``. An apply's record names the step's digest and
what moved. See :ref:`reference-audit-trail`.

.. seealso::

   :ref:`reference-trajectory-service`
      Every route, body and refusal code.

   :ref:`reference-trajectory-sequence`
      What a facility's sequence module holds, and the limits its commands
      meet.

   :ref:`architecture-trajectory-service`
      How the doors fit, whose action an action is, and what it passes.
