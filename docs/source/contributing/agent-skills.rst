Agent Skills
============

Osprey ships six installable **agent skills** --- packaged, step-by-step
instructions that a coding agent picks up automatically when a task matches
their description. Instead of re-explaining the contribution workflow or the
release process in every session, you install the skill once and the agent
follows the project's own playbook.

Install a skill globally (``~/.claude/skills/``) or into a single repository:

.. code-block:: bash

   uv run osprey skills install <name>                             # global
   uv run osprey skills install <name> --target .claude/skills/    # this repo only

The install mechanics --- backups of prior versions, ``--target`` resolution,
and the authoritative one-line roster --- are in the
:doc:`CLI reference </reference/cli>` under ``osprey skills``.

Skills for contributors
-----------------------

Four skills cover the contributor journey, in the order you meet them:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Skill
     - Install when
   * - ``osprey-design-philosophy``
     - You are designing, adding, or reviewing a feature. Encodes Osprey's
       design and architecture principles and the anti-pattern each one
       prevents --- consult it before adding a config knob, a new abstraction,
       or anything touching hardware-write safety. See
       :doc:`development-setup`.
   * - ``osprey-pre-commit``
     - You are ready to commit, push, or open a PR. Runs the three-tier
       check scripts (quick / ci / premerge) at the right gate.
   * - ``osprey-contribute``
     - You are taking a working-tree change to a merged PR. Walks through
       branching, atomic commits, push, PR, and CI iteration, auto-detecting
       whether you have push access to ``als-apg/osprey`` or work from a
       fork. The full workflow it follows is :doc:`workflow`.
   * - ``osprey-release``
     - You are a maintainer cutting a CalVer release: the release-notes PR,
       the tag, and verifying the automated PyPI publish.

The skills compose: each one routes the agent to its neighbours for adjacent
tasks, so ``osprey-contribute`` defers to ``osprey-pre-commit`` for a
standalone validation run and to ``osprey-release`` for cutting a release.

Skills for deployers and extenders
----------------------------------

The remaining two serve neighbouring audiences and are documented where
their subject lives:

- ``osprey-build-interview`` --- sets up or migrates an Osprey deployment
  through a guided conversation. This is a deployer's tool, not a
  contributor's; it has its own page at
  :doc:`/getting-started/osprey-build-interview`.
- ``creating-an-osprey-panel`` --- authors a themed web-terminal panel that
  passes the panel validator. The panel contract is on
  :doc:`/how-to/web-terminal/panels`, and the extension seam is on
  :doc:`extending-osprey`.
