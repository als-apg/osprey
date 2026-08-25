.. _contributing-workflow:

=======================
Git and GitHub Workflow
=======================

How a change travels from a local branch to a merged pull request on ``main``.

Branch Strategy
^^^^^^^^^^^^^^^

Osprey follows **GitHub Flow**: a single long-lived branch (``main``) with
short-lived topic branches that PR back into it. Releases are CalVer tags
(``vYYYY.M.P``) on ``main`` — no separate release branch.

**What this means for contributors:**

- Branch your work off ``main``, and open your PR against ``main``.
- ``main`` is always the integration target. CI gates every PR; protected status checks must pass before merge.
- Releases are cut by maintainers tagging a commit on ``main``; the PyPI publish workflow runs on ``v*.*.*`` tags.
- The ``osprey-connectors`` workspace package versions with the framework's calendar stream and releases from the **same** ``v*.*.*`` tag: one tag publishes both wheels, always carrying the same number. (Its old independent ``osprey-connectors-v*`` line is retired.)
- Hotfixes follow the same path: branch from the tag (or ``main``), PR back, tag again as ``vYYYY.M.P+1``. No special hotfix branches.
- Documentation publishes on the same tags: the site root is always the newest release; every push to ``main`` rebuilds ``/latest/``, shown with a development banner.

Branch Naming
^^^^^^^^^^^^^

- ``feature/description`` -- New features
- ``fix/description`` -- Bug fixes
- ``docs/description`` -- Documentation
- ``refactor/description`` -- Code refactoring
- ``test/description`` -- Test improvements

Making Changes
^^^^^^^^^^^^^^

**1. Create a branch:**

.. code-block:: bash

   git checkout -b feature/your-feature-name

**2. Make changes** -- follow the code standards below, add tests, update docs.

**3. Test locally** using the three-tier system:

.. code-block:: bash

   # Tier 1: Quick check (< 30s) -- before every commit
   ./scripts/quick_check.sh

   # Tier 2: Full CI check (2-3 min) -- before pushing
   ./scripts/ci_check.sh

   # Tier 3: Pre-merge check -- before creating a PR (compare against your PR target)
   ./scripts/premerge_check.sh main

**4. Commit changes** using conventional commit format:

.. code-block:: bash

   git add .
   git commit -m "feat(scope): short description

   - Detail about what changed
   - Another detail"

Commit Message Format
^^^^^^^^^^^^^^^^^^^^^

- ``feat:`` -- New features
- ``fix:`` -- Bug fixes
- ``docs:`` -- Documentation
- ``refactor:`` -- Code refactoring
- ``test:`` -- Tests
- ``chore:`` -- Dependencies, build

Every commit needs a corresponding CHANGELOG entry added **before** committing.

Pull Request Process
^^^^^^^^^^^^^^^^^^^^

1. Push your branch: ``git push origin feature/your-feature-name``
2. Open a PR on GitHub with a description, related issues, and testing performed.
3. PR requirements: pass all required CI checks, include a ``CHANGELOG.md`` entry for any user-visible change, and add appropriate tests. Internal-mode contributors with push access self-merge after CI is green (the ruleset does not require human approval); fork-mode contributions wait for a maintainer to merge.
4. During review: respond to feedback promptly, make requested changes, ask questions if unclear.

Branch Protection on ``main``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct pushes to ``main`` are rejected. All changes land via PR. The ruleset
enforces:

- All required CI checks must pass (no admin bypass).
- Linear history (use ``gh pr merge --rebase``; merge commits are rejected).
- Force-pushes and branch deletion on ``main`` are denied.

If a required check turns out to be wrong, fix it forward — there is no
escape hatch.

Dependency Update Pull Requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dependabot opens pull requests for dependency bumps, subject to a **seven-day
cooldown**: a release is not proposed until it has been public for a week, so
that a hijacked or malicious version has time to be found and yanked before it
reaches this repository. Security updates are exempt, by design — a fix for a
known vulnerability should not wait.

These pull requests run with a deliberate gap in coverage. GitHub treats a
Dependabot-triggered run as if it came from a fork: it receives only the
Dependabot secret store, never the repository's Actions secrets. The lanes that
need a live model endpoint therefore **skip** rather than run — the agentic
flows, the E2E suite, the dispatch stacks, and the two chat bridges. The run
summary of the ``All CI Checks Passed`` job names them explicitly, so a green
check on a dependency PR is never mistaken for full coverage.

To close that gap before merging, review the diff and then revalidate the branch
yourself. Because *you* trigger it, that run gets the normal secrets:

.. code-block:: bash

   gh workflow run ci.yml --ref <dependabot-branch> -f revalidate_secret_lanes=true

Find the branch name with ``gh pr view <number> --json headRefName``. Watch the
resulting run to completion before merging.

The reason this is a manual step rather than an automatic one is worth stating:
mirroring the model API key into the Dependabot secret store would make these
lanes pass unattended, but it would also hand a live credential to a
newly-published third-party package at install time — the precise supply-chain
exposure the cooldown exists to reduce. The human read of the diff is the point,
not an inconvenience around it.

Osprey Agent Workflow Skill
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use the Osprey agent (e.g., via `Claude Code <https://docs.claude.com/en/docs/claude-code>`_),
install the bundled ``osprey-contribute`` skill to get guided help following
this workflow:

.. code-block:: bash

   uv run osprey skills install osprey-contribute

The skill walks you through branching, commits, push, PR, and CI iteration,
auto-detecting whether you have push access to ``als-apg/osprey`` or are
contributing from a fork. It is one of six installable skills --- which one
fits which step of the journey is on :doc:`agent-skills`.
