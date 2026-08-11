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

   - **Metadata** — the plan's name, a human description, its category,
     which kinds of devices it needs, and whether it writes to the machine.
   - **Parameters** — a schema describing the knobs (names, types, limits).
     This is what BLUESKY's Plans view turns into a form, so a
     well-described parameter becomes a well-labeled field.
   - **The plan function** — builds the actual Bluesky plan from the
     parameters and the resolved devices.

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

.. seealso::

   :doc:`queue`
      How a queued plan actually runs, and what refusals mean.

   :doc:`/how-to/build-profiles`
      The build profile that owns ``plan_dir`` and ``excluded_plans``.
