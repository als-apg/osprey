.. _how-to-code-review:

=======================
Review the Agent's Code
=======================

OSPREY's safety checks read the agent's Python by its spelling: known write calls, known
dangerous patterns. Code review adds a reader that judges what the code would do. A second
AI agent reads every Python run the OSPREY agent sends to the executor, and every code cell
it writes into a notebook, following instructions you write, and says whether the code is
safe to run. You decide whether unsafe code is stopped or only reported. The agent gets the
reasons; you get a line in the activity log. :ref:`architecture-safety-chain-code-review`
explains how it fits with the other checks. It is off until you turn it on.

Turn it on
==========

Add the section to your profile. The two hooks have to be in the ``hooks:`` list as well;
``osprey build`` refuses to turn code review on without them:

.. code-block:: yaml

   code_review:
     enabled: true
     level: block          # stop unsafe code; `flag` only reports it

   hooks:
     - code-review         # reviews the notebook cells the agent writes
     - memory-guard        # keeps those edits inside the agent-data root

Run ``osprey build`` and reload the session. The reviewer uses the same model and provider
as the OSPREY agent and needs no setup of its own.

Write your rules
================

Every preset puts the reviewer's instructions at ``data/code_review/reviewer.md`` in your
project. The general part is OSPREY's: how to read control-system code, what to look for in
a write. The *Facility rules* section at the end is yours, with three placeholders to
replace:

- channels and subsystems the agent must never write, such as protection and safety
  systems;
- operating states or times when writes need extra care, such as during user beam;
- procedures or limits a write in an area must follow, and where they are documented.

The reviewer cannot look anything up, so a rule has to say what it needs: the channel name
as the agent spells it, the limit, the state. The agent cannot edit this file, and a change
to it counts from the next review on, with no rebuild.

Blocking or advisory
====================

The reviewer answers *pass*, *concern* or *block*. What a block does is your choice:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - ``level``
     - The reviewer says *block*
     - The reviewer could not answer
   * - ``block`` (default)
     - The code is stopped; the agent revises it and tries again
     - The code is stopped; the agent is told to report the problem, not retry
   * - ``flag``
     - The code runs; the agent is told what was found
     - The code runs unreviewed; the agent is told

A *concern* never stops code at either setting; it comes with the reasons. Start with
``flag`` while you tune your rules and watch the activity log, then switch to ``block``. By
default only read-write runs are reviewed; ``modes: [readonly, readwrite]`` reviews read-only
runs too. Notebook cells the agent writes are always reviewed when code review is on. One
review may take ``time_limit`` seconds (60 by default); that is the one key that needs
``osprey build`` and a reload to change. :ref:`profile-code-review` lists all keys.

What you see
============

On a Python run the review comes before the approval prompt: code the reviewer blocks in
blocking mode never reaches you, and otherwise the answer and the reasons are on the prompt
and in the approval preview, so you decide knowing what it found. Whenever the reviewer found
something, a *code review* line also appears in the activity log (under **⋯ More** in the web
terminal): what was reviewed, the answer, whether the code ran, and what the run was for. The
reasons go to the agent, in its conversation, where it was told to revise. Every review, a
pass included, leaves an entry in the audit trail with reason
``code_review`` (:ref:`reference-audit-trail`), so the record shows that the code was read
even when nothing was found.

When the reviewer is down
=========================

A review fails when the model cannot be reached or the time limit runs out. With
``level: block`` the agent's code is stopped and the agent reports the problem to you; the
fix is on your side, the provider or the limit, not in the code. Switching to ``level: flag``
or ``enabled: false`` in the rendered ``config.yml`` takes effect on the next call, so the
agent can keep working while you fix it. A ``code_review`` section that cannot be read at
all stops the code whatever the level says, because the level is part of what could not be
read; ``osprey build`` refuses such a section before it gets that far.
