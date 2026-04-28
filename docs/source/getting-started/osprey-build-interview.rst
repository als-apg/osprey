======================================
Guided Project Setup (Build Interview)
======================================

If you're setting up OSPREY for a specific detector, beamline, or accelerator
subsystem, the **build interview** walks you through a guided conversation that
generates a ready-to-build project profile tailored to your system. It also
handles **migration from existing OSPREY projects** — point it at your old
project directory and it will scan, classify, and extract everything reusable.

The whole interview takes about 10--15 minutes.

.. dropdown:: **Prerequisites**
   :color: info
   :icon: list-unordered

   **Required**

   * **OSPREY installed** — follow :doc:`installation` if you haven't yet.
   * **Claude Code** — the interview runs inside a Claude Code session via the
     ``/osprey-build-interview`` command. Install it from
     `claude.ai/code <https://claude.ai/code>`_ and make sure ``claude --version``
     works in your terminal.
   * **An Anthropic API key** (or any provider Claude Code is configured to use) —
     the interview is a live LLM conversation.

   **Recommended**

   * **A container runtime (Docker or Podman)** — not needed for the interview
     itself, but your generated project will likely include containerized
     services (Jupyter, simulation IOCs, databases). Without one, ``osprey build``
     still works but ``osprey deploy up`` won't. See the "Container Runtime"
     dropdown in :doc:`installation` for install instructions.
   * **A list or spreadsheet of EPICS PV names** for your subsystem, if you have
     one. Not required — the interview can proceed without concrete PVs — but
     having it handy speeds things up considerably.
   * **If migrating from an existing OSPREY project:** the path to that project
     directory ready to paste.

Install the interview skill
===========================

Install the interview skill with the OSPREY CLI:

.. code-block:: bash

   osprey skills install osprey-build-interview

This copies the skill into ``~/.claude/skills/osprey-build-interview`` and makes the
``/osprey-build-interview`` command available in any Claude Code session. Re-running
the command preserves your previous copy under
``~/.claude/skills/osprey-build-interview.bak.<timestamp>``.

Run the interview
=================

Create a working directory for your project and start Claude Code:

.. code-block:: bash

   # skip-ci
   mkdir -p ~/my-osprey-project
   cd ~/my-osprey-project
   claude

In the Claude Code session, type:

.. code-block:: text

   /osprey-build-interview

Claude will walk you through:

1. What system you work with and what you need the AI for
2. Whether you're starting fresh or **migrating from an existing OSPREY project**
   (if migrating, just point it to the directory and it will scan and reuse what it can)
3. Your EPICS PV names (if you have them — it's OK if you don't yet)
4. Whether you need read-only or write access
5. How to connect (simulated data is recommended for starting out)
6. Whether you'd like a custom monitoring panel in the web dashboard
7. A review step that checks for anything missing

Tips during the interview
-------------------------

- If you're not sure about a question, say "I'm not sure" — it'll pick a safe default
- If you have a spreadsheet of PV names handy, that's helpful but not required
- If you're migrating, have the path to your existing project directory ready
- You can always re-run the interview later to adjust things

Build your project
==================

When the interview is done, Claude generates a ``build-profile/`` directory
containing your ``profile.yml``, channel database, README, and a project-local
copy of the **osprey-build-deploy** skill under
``build-profile/.claude/skills/osprey-build-deploy/``. The interview installs
this skill automatically and runs three verification agents to confirm it
landed correctly — so the deploy phase is wired up by the time you see the
final summary.

Then:

.. code-block:: bash

   # skip-ci
   osprey build my-project build-profile/profile.yml

One command. OSPREY reads your profile, validates your selections, copies your
channel database into the right place, and produces a ready-to-use project.

To start using it:

.. code-block:: bash

   # skip-ci
   cd my-project && claude

Or for the web dashboard:

.. code-block:: bash

   # skip-ci
   osprey web

Phase 2: deploy your project
============================

The ``build-profile/`` directory is a durable, git-tracked artifact you'll
redeploy from many times. When you're ready to ship to a real deploy server
(GitLab CI/CD, container registry, on-server containers), open Claude Code
**inside the profile repo** and trigger the deploy skill:

.. code-block:: bash

   # skip-ci
   cd build-profile
   git init && git add -A && git commit -m "Initial profile"
   claude

In the Claude Code session:

.. code-block:: text

   /osprey-build-deploy

The deploy skill walks you through:

1. A one-time deploy interview that captures site-specific values (GitLab
   host, deploy server, container runtime, ports, optional modules) and
   writes them to ``facility-config.yml``
2. Scaffolding the deploy infrastructure from that config (``docker-compose.yml``,
   ``.gitlab-ci.yml``, ``scripts/deploy.sh``, ``.env.template``)
3. Driving the GitLab pipeline (push → CI builds containers → manual release
   tag → ``deploy.sh`` on the server)
4. Post-deploy health checks and ongoing release operations

Because the skill lives **inside the profile repo** (not globally), every
operator who clones this repo gets the same deploy operator automatically —
no separate install step. To refresh the skill after upgrading OSPREY, run
from the profile repo root:

.. code-block:: bash

   # skip-ci
   osprey skills install osprey-build-deploy --target .claude/skills/

The previous copy is backed up to ``.claude/skills/osprey-build-deploy.bak.<timestamp>/``.

Send feedback
=============

After you've tested your project, you can send feedback to the OSPREY team by
starting a Claude Code session and typing ``/osprey-build-interview feedback``. It takes
about 30 seconds and helps us improve the process.

See :doc:`/how-to/build-profiles` for the full build profile reference.
