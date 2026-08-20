====================
Guided Project Setup
====================

If you're setting up OSPREY for a specific detector, beamline, or accelerator
subsystem, the ``/osprey-build-interview`` skill turns a guided conversation
into a working deployment repository tailored to your system — created up
front from a curated preset, then refined with you piece by piece.

A minimal setup takes a few minutes; how much further you go is up to you,
and you can stop, build, and resume at any point.

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

The interview creates your **deployment repository first** — one ``osprey
init`` from a working preset, in the first minutes of the conversation — and
then refines it with you in place. There is no questionnaire to survive before
something exists: the repo builds at every step, and you can ask to see it
running at any point.

Four things are always settled before the interview calls itself done:

* **Which AI service answers** — usually whichever one your lab provides, and
  a working key for it. "I'm not sure" is a fine answer; the interview helps
  you find out.
* **How it connects** — the built-in simulated accelerator to start with,
  which is the recommended choice if you're unsure, or a live connection to
  your control system.
* **Whether the assistant may change things, or only look** — read-only is
  the safe start. If you do want it to change values, you'll also settle
  which channels and what range is safe for each.
* **What the project is** — a short name, the facility it belongs to, and
  its timezone.

Everything else keeps the preset's curated defaults unless you bring it up.
The profile the interview edits lists every optional feature — including the
ones not switched on — as commented entries with explanations, so "what else
could this do?" is always answerable by reading the file, with or without the
interview.

Decisions land in an ``INTERVIEW.md`` at the repository root: what was chosen,
why, and what was deliberately deferred. Reopen the repository later and
invoke the skill again to resume exactly where you left off.

Tips during the interview
-------------------------

- If you're not sure about a question, say "I'm not sure" — it'll pick a safe default
- If you have a spreadsheet of channel names handy, that's helpful but not required
- Ask to see the assistant running whenever you're curious — the repo always builds

Migrating an existing project
=============================

If you already have an OSPREY project — even one from the LangGraph era —
point the interview at it. It scans the old project, salvages what carries
over (channel databases, data files, custom code, configuration values), and
walks you through each judgment call as a confirmation rather than a
question. The porting decisions are recorded in ``INTERVIEW.md`` alongside
everything else.

Build and run
=============

The interview leaves you inside a deployment repository whose
``profile.yml`` records everything that was decided:

.. code-block:: bash

   # skip-ci
   cd my-project
   osprey build     # render build/ from the profile
   osprey web       # web dashboard on your own machine

Or talk to the agent directly with ``osprey chat``. Adjust anything later by
editing ``profile.yml`` (every key carries its own explanation) and running
``osprey build`` again.

Phase 2: deploy your project
============================

The interview settles *what* to build. Running what was built — putting it on a
real machine and keeping it there — is the other half, and it lives in the same
place. The deployment repository is a durable, git-tracked artifact you'll
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
