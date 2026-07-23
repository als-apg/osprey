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
     still works but ``osprey deploy up`` won't. See the "Container Runtime"
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

When the interview is done, the Osprey agent generates a ``build-profile/``
directory containing your ``profile.yml``, a ``README.md`` explaining what was
decided and why, and — if you gave signal details — a channel database and the
safe operating ranges that go with it.

Before handing it over, the agent builds the profile itself and requires that
build to succeed. If something doesn't render, it fixes the profile and tries
again rather than passing you a broken one, so what you receive is known to
build.

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

The interview settles *what* to build. A separate skill, **osprey-build-deploy**,
covers *how to ship it*. At the end of the interview the Osprey agent points you
at it:

.. code-block:: bash

   # skip-ci
   osprey skills install osprey-build-deploy

The ``build-profile/`` directory is a durable, git-tracked artifact you'll
redeploy from many times. When you're ready to ship to a real deploy server
(GitLab CI/CD, container registry, on-server containers), open the Osprey agent
**inside the profile repo** and trigger the deploy skill:

.. code-block:: bash

   # skip-ci
   cd build-profile
   git init && git add -A && git commit -m "Initial profile"
   claude

In the Osprey agent session:

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

If you'd rather every operator who clones the profile repo get the deploy skill
automatically, install a copy into the repo itself:

.. code-block:: bash

   # skip-ci
   osprey skills install osprey-build-deploy --target .claude/skills/

The previous copy is backed up to ``.claude/skills/osprey-build-deploy.bak.<timestamp>/``.

See :doc:`/how-to/build-profiles` for the full build profile reference.
