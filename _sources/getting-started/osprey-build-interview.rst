====================
Guided Project Setup
====================

If you're setting up OSPREY for a specific detector, beamline, or accelerator
subsystem, the ``/osprey-build-interview`` skill walks you through a guided
conversation that generates a ready-to-build project profile tailored to your
system.

It's a short conversation — roughly five minutes, and about three rounds of
questions.

.. dropdown:: **Prerequisites**
   :color: info
   :icon: list-unordered

   **Required**

   * **OSPREY installed** — follow :doc:`installation` if you haven't yet.
   * **The Osprey agent CLI** — the interview runs inside an Osprey agent session via the
     ``/osprey-build-interview`` command. Install it from
     `claude.ai/code <https://claude.ai/code>`_ and make sure ``claude --version``
     works in your terminal.
   * **An Anthropic API key** (or any provider the Osprey agent is configured to use) —
     the interview is a live LLM conversation.

   **Recommended**

   * **A container runtime (Docker or Podman)** — not needed for the interview
     itself, but your generated project will likely include containerized
     services (Jupyter, simulation IOCs, databases). Without one, ``osprey build``
     still works but ``osprey up`` won't. See the "Container Runtime"
     dropdown in :doc:`installation` for install instructions.
   * **A list or spreadsheet of EPICS PV names** for your subsystem, if you have
     one. Not required — the interview can proceed without concrete PVs — but
     having it handy speeds things up considerably.

Install the interview skill
===========================

Install the interview skill with the OSPREY CLI:

.. code-block:: bash

   osprey skills install osprey-build-interview

This copies the skill into ``~/.claude/skills/osprey-build-interview`` and makes the
``/osprey-build-interview`` command available in any Osprey agent session. Re-running
the command preserves your previous copy under
``~/.claude/skills/osprey-build-interview.bak.<timestamp>``.

Run the interview
=================

Create a working directory for your project and start the Osprey agent:

.. code-block:: bash

   # skip-ci
   mkdir -p ~/my-osprey-project
   cd ~/my-osprey-project
   claude

In the Osprey agent session, type:

.. code-block:: text

   /osprey-build-interview

There is no fixed script. The Osprey agent asks questions in whatever order the
conversation takes, phrases them for the person in front of it, and follows up
where an answer needs more detail. By the end of the conversation it will have
established five things:

* **What your system is** — the kind of system, a short name for the project, a
  one-line description in plain English, and the facility it belongs to.
* **How it connects** — simulated data to start with, which is the recommended
  choice if you're unsure, or a live connection to your control system.
* **Which signals matter** — the process variables the assistant will work with,
  along with their units and typical ranges. If you don't have a list yet, a
  rough description is enough to start from.
* **Whether the assistant may change things, or only look** — read-only is the
  default and the recommended starting point. If you do want it to change
  values, you'll also be asked which signals and what range is safe for each.
* **Which AI service you have access to** — usually whichever one your lab
  provides. "I'm not sure" is a fine answer here too.

Tips during the interview
-------------------------

- If you're not sure about a question, say "I'm not sure" — it'll pick a safe default
- If you have a spreadsheet of PV names handy, that's helpful but not required
- You can always re-run the interview later to adjust things

Build your project
==================

When the interview is done, the Osprey agent creates a **facility repository** —
a git repository named after your facility, with your profile nested inside it:

.. code-block:: text

   my-facility/
     profile/       the source you own: profile.yml, data/, your secrets
     build/         where `osprey build` renders projects (kept out of git)
     ci-extra.yml   your own CI jobs; nothing ever regenerates this
     .gitignore

``profile/profile.yml`` records everything the interview decided. Beside it are
a ``README.md`` explaining what was chosen and why, and — if you gave signal
details — a channel database and the safe operating ranges that go with it.

Before handing it over, the agent builds the profile itself and requires that
build to succeed. If something doesn't render, it fixes the profile and tries
again rather than passing you a broken one, so what you receive is known to
build.

Then:

.. code-block:: bash

   # skip-ci
   cd my-facility
   osprey build

One command. OSPREY reads your profile, validates your selections, copies your
channel database into the right place, and renders a ready-to-use project into
``build/my-project/``. You never have to say where the output goes: a profile
nested in a facility repository always renders into that repository's
``build/``, from whichever directory you run the command.

To start using it:

.. code-block:: bash

   # skip-ci
   cd build/my-project && claude

Or for the web dashboard:

.. code-block:: bash

   # skip-ci
   osprey web

Phase 2: deploy your project
============================

The interview settles *what* to build. Running what was built — putting it on a
real machine and keeping it there — is the other half, and it lives in the same
place. The facility repository is a durable, git-tracked artifact you'll
redeploy from many times, and both halves of deployment live inside it.

First, the deployment coordinates go in the profile itself, under a ``deploy:``
block: the CI platform you use, the deploy host, and the container registry if
that host pulls its images. A fresh profile ships this block commented out, so
filling it in is the one edit that turns a buildable profile into a deployable
one. Credentials are *named* there, never written there.

Second, one command turns those coordinates into files:

.. code-block:: bash

   # skip-ci
   osprey scaffold ci

That writes the CI pipeline at the repository root and a post-deploy health
check inside the profile. Re-run it whenever the ``deploy:`` block changes — a
file whose content already matches is left untouched, and a file you hand-edited
is reported rather than overwritten.

From there the same handful of commands runs the stack, from anywhere inside
the repository:

.. code-block:: bash

   # skip-ci
   osprey up -d      # start it
   osprey status     # what is running, where it answers, which build it is
   osprey logs       # what the containers are saying
   osprey health     # diagnostics: config, environment, providers, telemetry

``osprey status`` only reads, so it is safe against a live stack at any time,
and it is the first thing to run when something looks wrong. ``osprey down``
stops the stack and keeps its volumes.

See :doc:`/how-to/deploy-a-facility` for a worked example that goes from an
empty directory to running containers, and :doc:`/how-to/build-profiles` for the
full build profile reference.
