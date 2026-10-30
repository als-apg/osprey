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
- The ``osprey-connectors`` workspace package has its **own** CalVer stream, on its own ``osprey-connectors-vYYYY.M.P`` tags (``bN`` / ``rcN`` for pre-releases), cut when the package changes. Each release stays backward-compatible with every supported framework release, enforced in CI by the ``connectors-compat`` and ``framework-at-connectors-floor`` lanes, so the framework declares only a floor on it, never a pin. (The ``osprey-connectors-v0.1.0`` tag predates the scheme and is retired; the describe glob skips it.) A fork or mirror must sync the ``osprey-connectors-v*`` tags as well as the ``v*`` ones: without them the package builds as ``0.0.postN`` and ``osprey up --dev`` refuses it. Connectors releases are stable by default: a beta framework installs with pre-releases admitted and would absorb a connectors beta. A connectors tag runs its own publish workflow, ``release-connectors.yml``, and nothing else: the framework's ``v*.*.*`` release never builds the package, and resolves its ``osprey-connectors>=`` floor from PyPI, so a floor bump lands only after the connectors release it names is published.
- A change to what ships in the connectors wheel carries its changelog fragment in ``packages/osprey-connectors/changelog.d/``, which feeds that package's own ``CHANGELOG.md``; every other shipped change carries one in the root ``changelog.d/``. The README in each of those directories says which files need a fragment, and the changelog check names them in every verdict.
- ``packages/dock-engine`` is not a workspace member and publishes nowhere: the framework wheel ships its Python module ``dock_engine``, CANOPY installs the directory from git, and ``@als-apg/dock-engine`` is a git dependency. ``package.json`` stays at ``0.0.0-dev``. See :doc:`/reference/contracts/dock-engine`.
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

**3. Test locally** with the check command, ``scripts/check.py``, which runs
the merge gate from the checks every owner declares:

.. code-block:: bash

   # The quick tier -- before every commit (--fix lets ruff and prettier rewrite
   # files first)
   uv run python scripts/check.py quick

   # The full tier -- before pushing or opening a PR
   uv run python scripts/check.py full

   # The quick tier scoped to your diff (name your PR target; add --worktree
   # to judge uncommitted changes too)
   uv run python scripts/check.py quick --changed origin/main

``uv run python scripts/check.py list`` prints what each tier runs. The scoped
run also names the CI lanes your diff touches. ``--base BASE`` arms the checks
that judge a range without scoping the tier, so ``full --base origin/main``
judges the whole tree and the range in one verdict. Before any check runs, the
command puts the tree's pins in place: the pinned Claude Code first on ``PATH``
for the test sweep, and ``npm ci`` when ``node_modules`` differs from
``package-lock.json``. When it cannot, it refuses and names what to install.

A check is declared by the code that owns it, in a ``repo_checks.py`` module
beside that code: ``scripts/repo_checks.py`` holds the framework's and the
repository-wide checks, and a package that brings checks of its own, such as
``packages/osprey-connectors/``, carries its own module. ``scripts/check.py``
collects every such module in the tree; it holds no check itself. Each record in
a module's ``CHECKS`` carries everything the command does with the check -- its
command, scope, tiers, needs, bound and hooks -- so adding a check is one record
in its owner's module and a CI step that calls ``scripts/check.py run ID``.

JavaScript (``*.js`` and ``*.mjs``) is formatted by Prettier, exact-pinned in
``package.json`` and configured by ``.prettierrc.json``: ``npm run format``
rewrites the tree, and the ``prettier`` check (``npm run format:check``) fails
any file it would rewrite. Markdown, YAML, JSON, CSS and HTML are not formatted.
Type a value at its declaration, as a typed local or a typed constant, never
with a cast mid-expression: the formatter moves the parentheses such a cast
depends on, and the cast then types a different expression.

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

Every pull request that touches a directory the changelog gate covers needs a
changelog fragment: a small file in ``changelog.d/`` named ``<name>.<type>.md``
(use the issue number as the name when there is one).
``uv run python scripts/changelog_fragments.py check --base origin/main`` names the
directories the gate covers in every verdict. The ``lint`` CI job checks for a fragment,
and ``uv run python scripts/check.py quick --changed origin/main`` checks the same thing locally. Fragments
are folded into ``CHANGELOG.md`` when a release is cut — never add entries to
``CHANGELOG.md`` by hand. See ``changelog.d/README.md`` for the type list.

Pull Request Process
^^^^^^^^^^^^^^^^^^^^

1. Push your branch: ``git push origin feature/your-feature-name``
2. Open a PR on GitHub with a description, related issues, and testing performed.
3. PR requirements: pass all required CI checks, include a changelog fragment in ``changelog.d/`` when the changelog gate asks for one, and add appropriate tests. Internal-mode contributors with push access self-merge after CI is green (branch protection requires zero approving reviews); fork-mode contributions wait for a maintainer to merge.
4. During review: respond to feedback promptly, make requested changes, ask questions if unclear.

Branch Protection on ``main``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct pushes to ``main`` are rejected. All changes land via PR. GitHub branch
protection on ``main`` enforces:

- The required status checks must pass: ``pre-commit.ci - pr`` and
  ``All CI Checks Passed`` (the aggregate job every CI lane feeds). Admins
  cannot bypass them.
- Force-pushes and branch deletion on ``main`` are denied.

Linear history is not enforced: a PR may land as a merge commit or a rebase,
and both appear in ``main``'s history.

However a branch catches up with ``main``, its conflicts are resolved by
principle 8 of the :ref:`design principles <contributing-design-principles>`:
*where two lines of work meet, design from the end product*.

If a required check turns out to be wrong, fix it forward — there is no
escape hatch.

Model-Spending Lanes: ``full-ci`` on a PR, On Demand on ``main``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Eight CI lanes drive real agent sessions against a live model endpoint: the
agentic per-preset flows, the E2E suite, the two dispatch stacks, the scan-stack
and control-target-switch agentic lanes, and the two chat bridges. Together they
cost about $15 per run — and they used to run on every push of every pull
request, which at this repository's pace was the group's whole model bill.

They now run on exactly two events:

- **A pull request carrying the ``full-ci`` label.** Opt in per PR, for a change
  that touches what these lanes cover:

  .. code-block:: bash

     gh pr edit <number> --add-label full-ci

  Applying the label starts a fresh run of the PR's workflow (the workflow
  subscribes to the ``labeled`` activity); the superseded run is cancelled.
  The ``All CI Checks Passed`` run summary of every unlabeled PR names the
  lanes that skipped and prints this command.
- **The ``revalidate_secret_lanes`` dispatch** against any ref: the maintainer
  path described under Dependency Update Pull Requests below, and the way the
  lanes run against ``main`` at all:

  .. code-block:: bash

     gh workflow run ci.yml --ref main -f revalidate_secret_lanes=true

A release-notes pull request always carries ``full-ci``: these lanes are the
only proof the agentic paths work on the tree about to be tagged.

There is deliberately no schedule. A scheduled run spends model tokens with
nobody watching, and its red reaches nobody; a lane that only runs when someone
asked for it has a person waiting on the result. If you consume ``main``
directly, pin a sha a full run was green on rather than raw head.

Everything else on a PR is unchanged: the unit, lint, docs and package lanes and
every secret-free e2e lane (build-and-boot, deploy, bluesky, auth, podman) still
run on every push. Those cost runner minutes, not tokens, and the deploy lanes
are the per-PR proof that a render builds and starts.

Path filtering was measured and rejected as the mechanism: over the 161 PRs
merged in the two weeks before this landed, 86–91 % touched something under
``src/`` that the broad lanes cover, so a filter map would have saved about a
fifth and needed maintaining. The label is the whole mechanism; the stop-loss
behind it is a daily budget on the CI gateway key.

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

OSPREY Agent Workflow Skill
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use the OSPREY agent (e.g., via `Claude Code <https://code.claude.com/docs/en/overview>`_),
the ``/osprey:contribute`` skill gives guided help following this workflow. It
walks you through branching, commits, push, PR, and CI iteration,
auto-detecting whether you have push access to ``als-apg/osprey`` or are
contributing from a fork. It ships in the ``osprey`` plugin alongside five
others; how to install it, and which skill fits which step of the journey, is
on :doc:`agent-skills`.
