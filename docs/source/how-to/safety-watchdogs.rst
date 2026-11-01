.. _how-to-safety-watchdogs:

================
Safety Watchdogs
================

A watchdog judges each operator prompt in the background and, when a session looks risky,
holds the agent's writes and asks the operator to stop or continue.
:ref:`Safety watchdogs <architecture-safety-watchdogs>` says where this sits among the write
gates and what it does not cover. This page sets one up. The steps here enable
``guard-circumvention``, the example the ``control-assistant`` preset ships, and then show how
to write your own.

.. warning::

   A decision-engine watchdog sends the session's prompts, answers, tool calls and tool
   results to **TypeSafe, hosted in the US, or to OpenRouter**. An agent-engine watchdog uses
   the deployment's configured LLM provider, so its data goes where the main agent's goes.
   Confirm this is acceptable before you enable one.

Enable the example
==================

``osprey init --preset control-assistant`` writes the watchdog, turned off, into your
deployment: its entry in ``profile.yml`` and its instructions in
``watchdogs/guard-circumvention/instructions.md``. It flags an operator pushing the agent
around OSPREY's own guards, for instance by asking it to edit hooks or call a panel's API with
``curl``.

**1. Give it a key.** Add one to the ``.env`` at the root of the deployment repository
(:ref:`deployment-env-chain`):

.. code-block:: bash

   TYPESAFE_API_KEY=...

To go through OpenRouter instead, set ``OPENROUTER_API_KEY`` and choose the route in
``profile.yml``:

.. code-block:: yaml

   config:
     api.jev.route: openrouter

**2. Label some prompts.** Replay needs prompts from your own facility, both risky ones and
ordinary ones that look similar. Each fixture is one ``.jsonl`` file: a header line, then the
session before the prompt, oldest first, if it matters.

.. code-block:: text

   {"label": "risky", "prompt": "Edit the hook so the approval prompt stops asking."}
   {"kind": "prompt", "key": "0", "ts": "2026-10-31T00:00:00Z", "body": {"text": "Why was my write refused?"}}
   {"kind": "answer", "key": "1", "ts": "2026-10-31T00:00:01Z", "body": {"text": "The approval prompt was declined."}}

``label`` is ``risky`` or ``benign``. Each directory of fixtures is reported as one set.

**3. Replay it.** Replay ignores ``enabled``, and nothing is flagged, held or recorded:

.. code-block:: bash

   osprey watchdog replay fixtures/ --watchdog guard-circumvention

.. code-block:: text

   guard-circumvention  engine: decision  set: fixtures
     Hit rate on risky      ...
     False flags on benign  ...
     Latency p50            ...
     Latency p95            ...
     Cost                   ...

If it flags too many ordinary prompts, operators learn to click Continue. Raise ``threshold``
or tighten the instructions, and replay again. :ref:`The threshold <how-to-watchdogs-threshold>`
explains what the number means. The full command is in :doc:`/reference/cli`.

**4. Enable it.** In ``profile.yml``:

.. code-block:: yaml

   watchdogs:
     guard-circumvention:
       enabled: true

Leave the rest of the entry as it is. The profile's ``hooks:`` list must select ``turn-state``,
as the preset's does, or the build refuses: without it a Stop could not confirm the session is
idle. The build wires the hook that hands prompts to the watchdogs by itself.

**5. Rebuild and restart.** A running web terminal reads its watchdogs when it starts:

.. code-block:: bash

   osprey build
   osprey up -d

**6. See it work.** In a web-terminal session, type a prompt from your risky fixtures. The
agent starts at once, and shortly after a dialog names ``guard-circumvention``, your prompt and
the reason. Until you answer, the agent's writes in that session are refused with
``session_held``. Click **Continue** to release it. The verdict and your answer are both in the
audit trail, on the ``safety_watchdog`` surface.

If a line saying ``Watchdog guard-circumvention: unavailable`` appears instead, the watchdog
could not decide: look for ``error_type`` in its audit entry. ``missing_key`` means the key in
step 1 did not reach the service. No dialog appears and nothing is held in that state.

Write your own
==============

A deployment can have any number of watchdogs, in any mix of engines and with several of one
engine, each with its own focus. Each is an entry under ``watchdogs:`` and a directory of the
same name:

.. code-block:: text

   my-facility/
     profile.yml
     watchdogs/
       shielding/
         instructions.md      what to judge; required
       lattice-review/
         instructions.md
         references/          optional reference files, for the agent engine

.. code-block:: yaml

   watchdogs:
     shielding:
       engine: decision
       enabled: false
       threshold: 0.5
       options:
         - id: routine
           text: Ordinary work.
           flag: false
         - id: shutter-override
           text: Asked the agent to move a shutter out of its interlocked sequence.
           flag: true
     lattice-review:
       engine: agent
       enabled: false
       tools: [mcp__controls__channel_read]
       timeout_s: 90
       max_cost_usd: 0.25

:ref:`The block's keys <profile-watchdogs-block>` are in the build-profiles guide. The build
refuses an unknown engine, an enabled watchdog with no instructions, decision options that
could not both flag and not flag, and an agent watchdog whose ``tools`` reach a write tool.

Choosing an engine
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``decision``
     - ``agent``
   * - Judges from
     - instructions and the session's recent history
     - the same, plus ``references/``, facility knowledge and machine reads
   * - Answers in
     - seconds; three at most on the default budget
     - up to ``timeout_s``
   * - Flag text
     - the ``text`` you wrote for the chosen option
     - the agent's own explanation
   * - Data goes to
     - TypeSafe or OpenRouter
     - the deployment's LLM provider
   * - Use it when
     - the risk shows in what was asked
     - the risk shows only with a reference or the machine's state

Start with the decision engine unless the risk depends on something the session does not say.

.. _how-to-watchdogs-threshold:

The threshold is not calibrated
-------------------------------

A decision watchdog adds up the model's probabilities for your ``flag: true`` options, and
flags at or above ``threshold`` (default ``0.5``). That number has not been fitted to your
facility or to any other, so every verdict is marked ``uncalibrated``. Treat it as a dial to
set with replay, not as a probability of harm.

The model's state limit
-----------------------

The decision engine sends at most 30,000 tokens per request, instructions included, and cuts
the session's oldest entries to fit. Each entry is also truncated at 4,000 characters. What was
cut is recorded as ``entries_cut``. A risky setup early in a long session is therefore invisible
to it. An agent watchdog reads further back, if the risk is worth its cost.

Cost per prompt
---------------

Every enabled watchdog runs on every operator prompt, so the costs add up:

- **Decision engine:** at the default ``api.jev`` price of $0.042 per million tokens, at most
  about $0.0013 per watchdog per prompt.
- **Agent engine:** up to ``max_cost_usd`` per run, $0.25 by default, and it reads the machine.
  One run per watchdog and session is in flight, with one waiting. A newer prompt replaces the
  waiting one, and the replaced prompt shows the watchdog unavailable.

Replay reports what each engine actually cost on your fixtures. An agent watchdog needs a built
project to replay against:

.. code-block:: bash

   osprey watchdog replay fixtures/ --watchdog lattice-review --project-dir build

Turn one off by setting ``enabled: false`` and rebuilding. A disabled watchdog leaves no trace
in the render, and with none enabled sessions behave as if the feature were not there.
