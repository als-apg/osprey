==============
Facility Rules
==============

The always-in-context tier of :doc:`Facility Knowledge <use-facility-knowledge>`
is the set of Markdown rules under the project's ``.claude/rules/`` directory.
They load into the main agent's context at the start of every session, the same
way ``CLAUDE.md`` does. Several of them carry facility-specific operating
context.


What the ``control_assistant`` preset ships
===========================================

.. list-table::
   :header-rows: 1
   :widths: 34 51 15

   * - Rule
     - Holds
     - Facility-specific
   * - ``facility.md``
     - Facility identity (name, type, mission) and a pointer to the on-demand
       knowledge tools.
     - Yes
   * - ``control-system-safety.md``
     - Which control protocol the facility runs (EPICS, Tango, OPC-UA, LabVIEW)
       and the required ``osprey.runtime`` write path.
     - Yes
   * - ``timezone.md``
     - The facility timezone used to interpret timestamps.
     - Yes
   * - ``safety.md``
     - Channel-write safety, tool confinement, and data-integrity rules.
     - Customizable
   * - ``error-handling.md``
     - Error taxonomy and response protocol for tool failures.
     - Customizable
   * - ``artifacts.md``, ``python-execution.md``, ``data-visualization.md``, ``workflows.md``
     - Generic operating rules — artifact reuse, code execution, plotting, and
       task planning/delegation.
     - No

A build profile can add rules of its own. Any rule without ``paths`` frontmatter
loads unconditionally at session start.


Changing a rule
===============

There are two ways to edit a rule.

**Edit the Markdown directly.** Each rule is a file under ``.claude/rules/``.
``facility.md`` is yours to edit — it is user-owned, and ``osprey claude regen``
never overwrites it. The framework-generated rules *are* re-rendered by
``osprey claude regen``; to keep an edit to one of those, claim it first so it
becomes user-owned:

.. code-block:: console

   $ osprey scaffold claim rules/safety
   $ osprey scaffold diff rules/safety     # compare yours vs the framework version

**Through the web terminal.** ``osprey web`` exposes the agent's ``.claude/``
files in the browser: edit a rule in the setup editor, or use the scaffold
gallery to override a framework-generated rule (which claims it for you). See
:doc:`web-terminal/operate`.

.. seealso::

   :doc:`use-facility-knowledge`
      How the always-in-context rules relate to the on-demand OKF bundle.

   :doc:`build-profiles`
      How a build profile overlays its own rules into a generated project.
