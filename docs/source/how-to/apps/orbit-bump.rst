.. _how-to-orbit-bump:

===================
Example: Orbit Bump
===================

The orbit bump app is the example app the ``demo`` preset ships.
It solves a local closed-orbit bump — move the beam at one or more BPMs by a
chosen offset and close the bump with the surrounding correctors — on the
session's control target, shows the plan, and applies it behind an approval.
The operator works in the **Orbit bump** panel; the agent uses two tools on the
same backend. Both see one plan per login, control target and model: the plan
the agent computes is the plan the panel shows and the plan the approval card
binds.

This page reads the app three ways: as the operator, as the agent, and as its
source — the worked example for :doc:`build-your-own-app`.

The preset lists it in ``profile.yml`` as ``orbit-bump: {}``; delete the entry
to turn it off. The model dropdown offers the ring models that solve a closed
orbit — a served ``pyat`` model not set to ``single_pass`` — so a facility
without one has nothing to pick. The app sets a bump and leaves it; to step a
bump through amplitudes and measure the orbit shift at each, use Bluesky's
``orbit_bump_sweep`` plan (:doc:`/how-to/bluesky/run-first-plan`).

From the panel
==============

**1. Pick the plane and model.** *Horizontal* or *Vertical*; the model
dropdown lists the ring models with a closed-orbit model. The chip beside it
shows the state of the model's orbit response matrix (ORM).

**2. Pick the BPMs.** Drag across the dots on the strip to add every BPM in
the range; click a dot to toggle it, ⇧-click to extend from the last one
picked. Or type in the picker: ``22``, ``21-23`` (a range by ring position),
``2*`` (a prefix). ⇧-drag, the wheel and a double-click zoom; ⌥-drag pans.
Right-click the strip for the same verbs; the ☰ menu lists the gestures.

**3. Set the offsets.** Each target is an offset **from golden**, in the
facility's BPM unit — not an absolute position and not a change from the
current reading. Set them with the amplitude and shape (*flat*, *peak*,
*cosine*) over the bump BPMs, by dragging a ◆ handle or the bump's grip
(moves every target), with ↑/↓ on a focused handle or row (⇧ ×10), or by
typing in the table. The nudge step (1, 10, 100 µm) is in the ☰ menu.

**4. Choose the corrector rule.**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Rule
     - Correctors
   * - ``auto``
     - one upstream of the first BPM, every corrector between the BPMs, two
       downstream of the last
   * - ``tight``
     - the MML signed-index rule ``[-1 1]``, relative to the bump BPMs
   * - ``manual``
     - the correctors you click on the strip

The ⋯ menu after the toggle adds two wider MML lists, ``[-2 -1 1 2]`` and
``[-3 -1 1 3]``.

**5. Compute.** Compute writes nothing. It replaces your current plan with a
new one and draws the predicted orbit and each corrector's step under the
strip. Editing the bump afterwards marks it *edited · not computed* and
holds Apply until the next Compute. A bump BPM that cannot be read, or reads
a non-finite value, blocks compute and is named.

**6. Read the card.** The Correctors card has one row per corrector: ``now``,
delta, ``new`` and the limit verdict; a row's note is its tooltip, and the
closure line is the tooltip of the card's title. The confirmation shows all of
it, with the channel each row writes.

- Verdict: the declared limit range, ``no declared limit``, ``limits checking
  off``, or a refusal with the overshoot (``over by …``). Nothing is clamped.
- Note: ``no tolerance: not staleness-checked`` marks a setpoint without a
  declared tolerance; ``not settled`` shows a readback that has not reached
  its setpoint.
- Closure line: singular values kept (``σ kept``), ``max predicted leakage``
  at the leakage BPMs, dropped leakage BPMs, and BPMs skipped because the ORM
  could not solve them.

**7. Apply.** *Apply…* is enabled only on a ``ready``, unedited plan with
writes on. It
opens a confirmation with the setpoints and the closure line; *Confirm* writes
the rows once, in plan order.

**8. Iterate.** After a full apply the plan is ``applied``, the orbit is
re-read, and the Correctors title's tooltip shows ``residual after`` at the
bump BPMs. To refine, compute again on the re-read orbit and apply the new
plan. Every write is on a confirmed or approved plan.

**Recompute** on the ORM chip rebuilds the orbit response matrix from the
model; nothing is measured. A new ORM discards the plans on that model not yet
applied (``ready`` and ``not_applicable``); compute again.

With writes off for the session's machine the panel names the reason
(*Writes are off*) and refuses apply; compute, discard and recompute still
work, so a bump can be planned and inspected. On a login that does not arm
the app the panel shows state and the approval only.

From the agent
==============

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tool
     - Does
   * - ``plan_bump(model, plane, targets, rule, correctors=None)``
     - Solves the bump; writes nothing; replaces the current plan and returns
       its ``plan_id``. ``targets`` maps BPM to offset from golden; ``rule``
       is ``auto``, ``mml:[…]``, or ``manual`` with ``correctors``.
   * - ``apply_bump(plan_id)``
     - Writes the plan. Runs only behind an approval bound to that plan id
       and its setpoints. A plan id that is not current is refused, naming
       the current plan.

Ask the agent to *move BPM 7 up by 0.2 mm in the vertical plane*. It calls
``plan_bump``, reports the plan — the correctors, each setpoint with its
limit verdict, the predicted residual — and then asks to apply. At *asked*
the approval card appears with the plan id and one line per corrector; the
operator's approval is the confirmation. At *allowed* the apply runs bound to
the same plan, with no card. If ``apply_bump`` gets no answer in time, it
reports the plan status and the row outcomes the backend holds rather than
retrying.

The app's skill tells the agent all of this — which tool for which request,
that offsets are from golden, to show the plan before asking to apply, what
each status and refusal means — so an operator can ask in the facility's words
and the agent uses the tools as the panel would.

How it is built
===============

The app is a ``demo`` deployment's ``apps/orbit-bump/``: the
three parts of :doc:`index`, with the manifest:

.. literalinclude:: ../../../../src/osprey/profiles/presets/apps/orbit-bump/app.yml
   :language: yaml

**The backend** (``orbit_bump/server.py``) keeps the plans in a store under
the container's ``state`` directory, one current plan per login, target and
model, so a plan survives a page reload and a restart. It knows no facility of
its own: the ring models, their BPMs and correctors and each model's response
matrix come from the SDK (``model_members``, ``model_response_matrix``), the
orbit and golden orbit are read through the connector, and the limit verdict
and tolerance of each setpoint come from the SDK's pre-write verdict
(``prewrite_verdict``), the same judgement the connector makes at the write. Four routes
are the agent's: ``GET /api/state`` (models, orbit, posture, the current
plan), ``GET /api/approval`` (what the card binds), ``POST /api/compute`` and
``POST /api/apply``. The panel's recompute and discard routes are not in
``agent_routes``, so the agent cannot call them.

**The tool server** (``orbit_bump/tools.py``) is two functions. ``plan_bump``
posts to ``/api/compute``. ``apply_bump`` runs ``check_approved_call``, takes
the plan id and setpoint digest the approval bound, refuses a ``plan_id`` that
differs from the approved one, and posts both to ``/api/apply`` with a timeout
long enough for the writes to settle and the orbit to be re-read — which is why
the manifest's ``read_timeout_s`` is 90.

**The approval entry** is the heart of the example. ``state_path`` is a route
made for the card, ``/api/approval``: it answers only while the current plan is
``ready`` and fits a card, with the plan id, a digest of the setpoints and
``display_lines``, one per corrector. The card shows those lines, so the
operator approves every setpoint, not a summary. ``bind.state_fields`` copies
the plan id and the digest into the call, and the backend applies only when
both still match the plan it holds. A plan recomputed after approval, or
approved on another machine, is refused without a write.

**The panel** (``panel/``) is the strip, the model and ORM chip, the plan card
and the confirm dialog, in plain JavaScript on the terminal's design tokens.
It reads the backend's posture and shows *Writes are off* rather than hiding
the controls. Its tests under ``tests/`` run with ``osprey app check``.

**The skill** (``skills/orbit-bump/SKILL.md``) is listed under ``skills:``
and installed into every login that shows the app.

``deny_in_jobs: true`` keeps every tool out of dispatch jobs: a bump is an
operator's decision. ``control_identity: true`` makes the container's writes
reach the control system under the person's name rather than the shared
``osprey`` account.

Plan states
===========

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Status
     - Meaning
   * - ``ready``
     - Can be applied.
   * - ``applying``
     - Being written; cannot be discarded until it ends.
   * - ``not_applicable``
     - A new setpoint violates a declared limit, or the rows do not fit the
       approval card. The card states the reason. Change targets or rule and
       compute again.
   * - ``superseded``
     - Replaced by a newer plan.
   * - ``discarded``
     - Dropped, for instance by a new ORM.
   * - ``applied``
     - Every row confirmed; the orbit was re-read.
   * - ``partially_applied``
     - A row did not confirm and the sequence stopped. Cannot be re-applied;
       compute a new plan from the current state.

Row outcomes
============

After an apply each row carries an outcome with its readback. Any outcome
other than ``confirmed`` stops the sequence.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Outcome
     - Meaning
   * - ``confirmed``
     - The connector confirmed the write.
   * - ``mismatch``
     - Written; the readback disagrees.
   * - ``refused``
     - The write was refused.
   * - ``failed``
     - The write failed.
   * - ``unconfirmed``
     - Hardware state unknown.
   * - ``not attempted``
     - An earlier row stopped the sequence.

Refusals
========

Nothing is written when an apply is refused.

- **Stale.** A corrector's readback moved since compute by more than the
  tolerance its setpoint declares. Compute again and approve the new plan.
  Rows marked ``no tolerance: not staleness-checked`` are not checked. Orbit
  motion since compute is not staleness.
- **Writes off.** With writes off for the session's machine, apply and
  ``apply_bump`` are refused and the panel names the reason; compute and
  ``plan_bump`` still plan. See :ref:`how-to-apps-writes`.
- **Unarmed login.** Reads state and approvals only; compute, discard and
  recompute are refused.
- **Owner, target or model differ** from the plan's: refused. Compute on the
  current target and model.
- **Approval missing or for another plan:** refused. Approve the current plan.
