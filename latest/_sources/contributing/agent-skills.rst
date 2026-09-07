.. _contributing-agent-skills:

============
Agent Skills
============

Osprey ships nine **agent skills** --- packaged, step-by-step instructions that a
coding agent picks up automatically when a task matches their description.
Instead of re-explaining the contribution workflow or the release process in
every session, you install them once and the agent follows the project's own
playbooks.

All nine travel together in one plugin, ``osprey``, published from the root of
the ``als-apg/osprey`` repository. Claude Code and Codex install it from there
with two commands each.

Install in Claude Code
----------------------

.. code-block:: bash

   claude plugin marketplace add als-apg/osprey --sparse .claude-plugin plugins
   claude plugin install osprey@osprey

Both commands need Claude Code 2.1.228 or newer.

The first command registers the marketplace. The second installs the plugin at
user scope, which is the default, so the skills are available in every session
that reads user settings.

``--sparse`` keeps the working checkout to the manifest and plugin directories.
It does not reduce how much git transfers.

Invoke a skill by its namespaced name, for example ``/osprey:contribute``.

.. warning::

   Registering a marketplace named ``osprey`` from a second source --- a local
   clone, say --- silently repoints the name to that source.

Inside a deployment repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the plugin a second time at project scope:

.. code-block:: bash

   claude plugin install osprey@osprey --scope project

Sessions launched by ``osprey chat`` and by the web terminal run with
``--setting-sources project``. Those sessions cannot see user-scope plugins
(verified 2026-09-01), so without the project-scope install the skills never
load there. The install writes ``enabledPlugins`` into the repository's
``.claude/settings.json``.

Updating
^^^^^^^^

.. code-block:: bash

   claude plugin marketplace update osprey
   claude plugin update osprey@osprey

The first command refreshes the marketplace listing; the second installs the
version it now points at. Restart the session to pick it up. Add
``--scope project`` to the second command to update a project-scope install.

Iterating on a skill locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From a checkout of the repository, load the plugin straight off the working
tree instead of installing it:

.. code-block:: bash

   claude --plugin-dir plugins/osprey

The session reads the skills from ``plugins/osprey/skills/``, so you can edit a
``SKILL.md`` and start a new session to try it.

Install in Codex
----------------

.. code-block:: bash

   codex plugin marketplace add als-apg/osprey --sparse .agents/plugins --sparse plugins
   codex plugin add osprey@osprey

Codex invokes a skill with a leading ``$``, for example ``$contribute``. The
skills themselves are the same files the Claude Code plugin serves.

.. note::

   Command syntax per `learn.chatgpt.com/docs/developer-commands
   <https://learn.chatgpt.com/docs/developer-commands>`_, retrieved 2026-09-01,
   and verified against codex-cli 0.149 on 2026-09-02.

The nine skills
----------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Skill
     - What it does
   * - ``/osprey:design-philosophy``
     - Osprey's design and architecture principles, and the anti-pattern each
       one prevents. Consult it before adding a config knob, a new
       abstraction, or anything touching hardware-write safety.
   * - ``/osprey:contribute``
     - Takes a working-tree change to a merged PR on ``main``: branch, atomic
       commits, push, PR, CI iteration. Detects whether you have push access
       or work from a fork.
   * - ``/osprey:pre-commit``
     - Runs the three-tier check scripts --- quick, ci, premerge --- at the
       gate that matches what you are about to do.
   * - ``/osprey:release``
     - Walks a maintainer through a CalVer release: the release-notes PR, the
       tag, and verifying the automated PyPI publish.
   * - ``/osprey:install``
     - The installer, as a conversation: installs Osprey if it is missing,
       inventories what exists, and sets up or migrates a deployment for an
       accelerator, beamline, or detector one confirmed card at a time.
   * - ``/osprey:upstream-scout``
     - Investigates whether something a facility needs is an Osprey gap, a
       deployment gap, or already supported; judges whether the fix is
       mechanical or architectural; drafts the write-up to file or to branch
       from. ``/osprey:install`` runs it in the background.
   * - ``/osprey:panel``
     - Authors a themed web-terminal panel that passes the panel validator.
   * - ``/osprey:housekeeping``
     - Finds where the generated project, the shipped prompts, the runtime
       messages, or a pinned version say something the code no longer does,
       proves it, and writes a report the maintainer rules on. Report only.
   * - ``/osprey:doc-sync``
     - Proves the pages under ``docs/source`` against the code, from reading
       source and from running what a page documents, reports the drift,
       and applies the doc-side fixes the maintainer accepts.

In Codex the same nine are ``$design-philosophy``, ``$contribute``,
``$pre-commit``, ``$release``, ``$install``, ``$upstream-scout``, ``$panel``,
``$housekeeping``, and ``$doc-sync``.

The skills route to each other: ``/osprey:contribute`` hands a standalone
validation run to ``/osprey:pre-commit`` and a release to ``/osprey:release``;
``/osprey:release`` runs ``/osprey:housekeeping`` and ``/osprey:doc-sync`` as
advisory steps before the release-notes PR, ``/osprey:housekeeping`` hands
doc-page items to ``/osprey:doc-sync``, ``/osprey:install`` launches
``/osprey:upstream-scout`` in the background, and the scout's branch path hands
the implementation to ``/osprey:contribute``.

The workflow behind each one is documented on its own page:

- :doc:`workflow` --- the contribution journey ``/osprey:contribute`` follows.
- :doc:`development-setup` --- the checks ``/osprey:pre-commit`` runs.
- :doc:`/getting-started/osprey-install` --- the deployer-facing page,
  written for someone standing up a deployment.
- :doc:`/how-to/web-terminal/panels` --- the panel contract, with the
  extension seam on :doc:`extending-osprey`.
