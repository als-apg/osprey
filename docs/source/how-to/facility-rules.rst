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

Answer provenance (verify-first)
================================

Beyond the ``.claude/rules/`` files, the agent's *answer posture* is set by two
framework-generated artifacts — the ``control-operator`` output-style and the
generated ``CLAUDE.md``. Both instruct the agent to answer **verify-first**: for
a factual question it queries the appropriate tool or source first and leads with
that result, naming the source; anything it cannot back with a tool is flagged
plainly and up front — never a confident lead with a buried caveat, and never an
answer followed by an optional offer to verify. A multi-tool or research answer
closes with an explicit provenance summary — the sources it used and a brief
confidence/scope note — while single reads stay terse.

Because it ships in framework-generated artifacts, the claimed-artifact caveat
from `Changing a rule`_ applies. A deployment that has ``osprey scaffold
claim``ed ``CLAUDE.md`` (``claude-md``) or the ``control-operator`` output-style
keeps its own copy and will **not** pick up this behavior on ``osprey claude
regen``. To adopt it, review the framework version and either merge it by hand or
unclaim and regen:

.. code-block:: console

   $ osprey scaffold diff output-styles/control-operator     # framework vs. yours
   $ osprey scaffold unclaim output-styles/control-operator  # then: osprey claude regen


.. seealso::

   :doc:`use-facility-knowledge`
      How the always-in-context rules relate to the on-demand OKF bundle.

   :doc:`build-profiles`
      How a build profile overlays its own rules into a generated project.
