====================
Guided Project Setup
====================

The ``/osprey:build-interview`` skill turns a conversation with the Osprey agent
into a deployment repository for your accelerator, beamline, or detector. It
starts from what you already have, agrees each step with you, and ends on a
project that validates and builds. You can stop and resume at any point.

.. dropdown:: **Prerequisites**
   :color: info
   :icon: list-unordered

   * **OSPREY 2026.9.0b1 or newer** — follow :doc:`installation` if you haven't
     installed it yet. Check with ``osprey --version``.
   * **The Osprey agent CLI** — the interview runs inside an Osprey agent session.
     Install it from `claude.ai/code <https://claude.ai/code>`_ and make sure
     ``claude --version`` works in your terminal.
   * **A provider API key** — the interview is a live conversation with an AI
     service, usually whichever one your lab provides.
   * **Recommended:** a container runtime (Docker or Podman) so that ``osprey up``
     works on the result, and a list of your channel names if you have one.

Install the interview skill
===========================

The interview ships in the ``osprey`` plugin:

.. code-block:: bash

   # skip-ci
   claude plugin marketplace add als-apg/osprey --sparse .claude-plugin plugins
   claude plugin install osprey@osprey

How to update the plugin or run it from a checkout is on
:doc:`/contributing/agent-skills`.

Run the interview
=================

.. code-block:: bash

   # skip-ci
   mkdir -p ~/my-osprey-project
   cd ~/my-osprey-project
   claude

In the Osprey agent session, type:

.. code-block:: text

   /osprey:build-interview

What happens
============

.. graphviz::
   :align: center

   digraph interview {
     rankdir=TB;
     graph [fontname="Helvetica", nodesep=0.3, ranksep=0.32, bgcolor="transparent"];
     node  [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled",
            fillcolor="#f5f7fa", color="#7a869a", fontcolor="#1f2933", margin="0.22,0.10"];
     edge  [color="#7a869a", arrowsize=0.8];

     start [label=<<b>What already exists?</b><br/>an OSPREY deployment · a facility, no OSPREY · nothing yet>, fillcolor="#e8eef7"];

     discover [label=<<b>DISCOVER</b><br/>Reads your repository, files and command output.<br/>Writes nothing.>];
     card1 [label="Status-quo card: what is there now\nyou confirm, or say what is wrong", shape=note, fillcolor="#fff8e1"];

     map [label=<<b>MAP</b><br/>One verdict per thing found: carry it over,<br/>use what OSPREY has, drop it, or flag a gap.<br/>Then asks for facility name, prefix, timezone, project name.>];
     card2 [label="Porting-map card: what carries over\nyou confirm, or say what is wrong", shape=note, fillcolor="#fff8e1"];

     build [label=<<b>BUILD</b><br/>Creates the project from the hello-world preset,<br/>then adds one confirmed piece at a time:<br/>AI provider · control system · writes on or off ·<br/>users and roles · facility knowledge stubs.<br/>Validates after every change.>];
     card3 [label="Gap card: what the full example has and you do not\nclose or skip each row, with a reason", shape=note, fillcolor="#fff8e1"];

     close [label=<<b>CLOSE</b><br/>A second agent argues against the setup.<br/>Final validate and build.>];
     done [label=<<b>Deployment repository</b><br/>profile.yml holds every decision<br/>INTERVIEW.md holds every confirmed card>, fillcolor="#e6f4ea"];

     start -> discover -> card1 -> map -> card2 -> build -> card3 -> close -> done;
   }

Every phase ends with a card, a short summary of what was concluded, and one
question: confirm, or say what should change. A confirmed card is written into
``INTERVIEW.md`` and is not reopened.

To resume, open the repository and run ``/osprey:build-interview`` again. It
continues after the last confirmed card.

Migrating an existing project is the same interview. Answer "an OSPREY
deployment already exists" and DISCOVER inventories it, whatever its generation.
The porting map is your migration plan. Files that carry over are copied
unchanged, and custom code that needs real work is recorded in ``INTERVIEW.md``
as work to do rather than attempted mid-conversation.

Tips during the interview
-------------------------

- If you're not sure about a question, say "I'm not sure". It picks a safe default
  and records that it did.
- Ask to see the assistant running whenever you're curious. You can ask for a
  build at any pause.

Build and run
=============

The interview leaves you inside a deployment repository:

.. code-block:: bash

   # skip-ci
   cd my-project
   osprey build     # render build/ from the profile
   osprey web       # web dashboard on your own machine

Or talk to the agent directly with ``osprey chat``. Adjust anything later by
editing ``profile.yml`` (every key carries its own explanation) and running
``osprey build`` again.

Deploy it
=========

Deployment coordinates go in the profile under a ``deploy:`` block: the CI
platform, the deploy host, and the container registry if that host pulls its
images. A fresh profile ships this block commented out. Credentials are named
there, never written there. Then:

.. code-block:: bash

   # skip-ci
   osprey scaffold ci    # CI pipeline + post-deploy health check script
   osprey up -d          # start it
   osprey status         # what is running, where it answers, which build it is

See :doc:`/how-to/deploy-a-facility` for a worked example from an empty directory
to running containers, and :doc:`/how-to/build-profiles` for the full build
profile reference.
