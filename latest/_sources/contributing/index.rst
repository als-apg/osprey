Contributing to Osprey
======================

Thank you for your interest in contributing to the Osprey Framework. This guide covers environment setup, Git workflow, code standards, and community guidelines.

----

Environment Setup
-----------------

**Prerequisites:** Python 3.11+, Git, a GitHub account, and `uv <https://docs.astral.sh/uv/>`_.

**1. Fork and Clone**

.. code-block:: bash

   git clone https://github.com/YOUR-USERNAME/osprey.git
   cd osprey

**2. Install Dependencies**

.. code-block:: bash

   # Install all dev and docs dependencies (creates .venv automatically)
   uv sync --extra dev --extra docs

   # Add a new dependency
   uv add <package>

**3. Set Up Pre-commit Hooks**

.. code-block:: bash

   pre-commit install

Hooks auto-fix formatting and prevent commits with common problems.

**4. Verify Installation**

.. code-block:: bash

   uv run pytest tests/ --ignore=tests/e2e -v

If all tests pass, you are ready to contribute.

----

Git and GitHub Workflow
-----------------------

Branch Strategy
^^^^^^^^^^^^^^^

Osprey follows **GitHub Flow**: a single long-lived branch (``main``) with
short-lived topic branches that PR back into it. Releases are CalVer tags
(``vYYYY.M.P``) on ``main`` — no separate release branch.

**What this means for contributors:**

- Branch your work off ``main``, and open your PR against ``main``.
- ``main`` is always the integration target. CI gates every PR; protected status checks must pass before merge.
- Releases are cut by maintainers tagging a commit on ``main``; the PyPI publish workflow runs on ``v*.*.*`` tags.
- Hotfixes follow the same path: branch from the tag (or ``main``), PR back, tag again as ``vYYYY.M.P+1``. No special hotfix branches.

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

Osprey Agent Workflow Skill
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use the Osprey agent (e.g., via `Claude Code <https://docs.claude.com/en/docs/claude-code>`_),
install the bundled ``osprey-contribute`` skill to get guided help following
this workflow:

.. code-block:: bash

   uv run osprey skills install osprey-contribute

The skill walks you through branching, commits, push, PR, and CI iteration,
auto-detecting whether you have push access to ``als-apg/osprey`` or are
contributing from a fork. It composes with the other bundled skills:

- ``osprey-pre-commit`` -- standalone validation runs
- ``commit-organize`` -- splits a messy working tree into atomic commits
- ``osprey-release`` -- the release-cutting flow for maintainers
- ``osprey-design-philosophy`` -- OSPREY's design and architecture principles,
  for designing or reviewing a feature before you open the PR

List all installable skills with ``uv run osprey skills install --help``.

----

Code Standards
--------------

Design Principles
^^^^^^^^^^^^^^^^^

Before designing a new connector, MCP server, provider, capability, or any
non-trivial feature, consult OSPREY's design and architecture principles -- the
safe-state default, facility-neutral core, measured symmetry with peer
subsystems, swappable components, and discoverable user-facing features.
Install the bundled skill so the Osprey agent applies them as you design and
review:

.. code-block:: bash

   uv run osprey skills install osprey-design-philosophy

The principles guide decisions; they are not mechanical rules. When a change
feels wrong but the reason is hard to name, they help you name the drift and
correct it before you open the PR.

Python Style
^^^^^^^^^^^^

We follow PEP 8 with Ruff enforcement:

- **Line length**: 100 characters
- **Type hints**: Gradual typing enforced with mypy
- **Docstrings**: Google style
- **Classes**: PascalCase, **Functions**: snake_case, **Constants**: UPPER_SNAKE_CASE

**Import organization:** standard library, then third-party, then local (``from osprey...``).

Linting and Formatting
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Lint and format
   uv run ruff check src/ tests/
   uv run ruff format src/ tests/

   # Auto-fix lint issues
   uv run ruff check --fix src/ tests/

   # Type checking
   uv run mypy src/

Testing
^^^^^^^

All new functionality must include tests.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Type
     - When to Use
     - Cost/Speed
   * - **Unit**
     - Pure functions, business logic, utilities
     - Fast, no external dependencies
   * - **Integration**
     - Component interactions, API endpoints
     - Medium
   * - **E2E**
     - Critical user flows, deployment validation
     - Slow, requires API keys ($0.10-$0.25/run)
   * - **Browser**
     - Real-browser page loads, theming, JS module loading (Playwright + Chromium)
     - Slow; needs Chromium (auto-installed in CI, skips locally if absent)

**Running tests:**

.. code-block:: bash

   # Unit tests (fast, no API keys required)
   uv run pytest tests/ --ignore=tests/e2e -v

   # Single test file
   uv run pytest tests/path/to/test_file.py -v

   # Single test function
   uv run pytest tests/path/to/test_file.py::test_function_name -v

   # E2E tests (requires API keys) -- MUST use path, NOT marker
   uv run pytest tests/e2e/ -v

   # Browser smokes (Playwright + Chromium; skips if the browser is absent)
   uv run pytest tests/interfaces/ -m browser -v

   # With coverage
   uv run pytest tests/ --ignore=tests/e2e --cov=src/osprey

.. warning::

   E2E tests **must** be run with ``pytest tests/e2e/`` not ``pytest -m e2e``.
   The marker-based approach causes registry state leaks and service conflicts.

Front-End (JavaScript) Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Front-end code (``static/js/`` under each ``src/osprey/interfaces/<name>/``) gets
its own dev/CI-only Node toolchain -- ``tsc --noEmit`` for types and
`Vitest <https://vitest.dev/>`_ for unit tests. Neither is needed to install or
run Osprey; both run only in dev and CI, the front-end analogues of ``mypy``
and ``pytest``.

.. code-block:: bash

   # Type-check every // @ts-check'd file (opt-in per file -- see below)
   npm run typecheck

   # All Vitest units (happy-dom, no real browser)
   npm run test:js

   # A single Vitest file
   npx vitest run tests/interfaces/artifacts/preview.test.mjs

A front-end change is covered by up to five rails, narrowest/fastest to
broadest/slowest:

.. list-table::
   :header-rows: 1
   :widths: 18 47 35

   * - Rail
     - Covers
     - Run
   * - **Vitest unit**
     - Pure logic in one module: mocked ``fetch``, no real browser (happy-dom)
     - ``npx vitest run tests/interfaces/<iface>/<module>.test.mjs``
   * - **Loads-clean**
     - Page boots in a real browser with no uncaught JS exception and no
       failed same-origin script/stylesheet fetch
     - ``uv run pytest tests/interfaces/test_load_smokes.py -m browser -k <iface> -v``
   * - **Contract**
     - Shell<->panel chrome contract: ``?embedded=true`` hides branding, the
       theme switcher shows/hides correctly, a reload after a theme toggle
       carries no stale ``?theme=``
     - ``uv run pytest tests/interfaces/web_terminal/test_contract_params.py -m browser -v``
   * - **Visual**
     - Pixel-level screenshot diff per interface x theme against a committed
       baseline PNG
     - ``uv run pytest tests/interfaces/design_system/test_visual.py -k <iface> -v``
   * - **Interaction pin**
     - A real user flow through a real browser and a real backend, proving a
       multi-module wiring didn't drop a call across a split -- the one net a
       per-module Vitest suite (which mocks its neighbors) cannot cast
     - ``uv run pytest tests/interfaces/<iface>/test_<flow>.py -m browser -v``

.. warning::

   ``test_visual.py`` is marked ``slow``, **not** ``browser`` -- an
   ``-m browser`` selector silently matches zero of its cases. Select it with
   ``-k <target-name>`` (or run the file directly) instead.

**What a new panel or extracted module must ship with:**

- ``// @ts-check`` as line 1, plus full JSDoc (``@param``/``@returns``).
  ``tsconfig.json`` sets ``checkJs: false`` repo-wide, so a file is
  type-checked only once it opts in -- a module missing the pragma makes
  ``npm run typecheck`` *pass* while checking nothing inside it.
- A Vitest file, one-to-one by name (``foo.js`` -> ``foo.test.mjs``), covering
  the module's pure logic and DOM-visible behavior in isolation.
- If the module is wired into a page: a ``test_load_smokes.py`` entry for
  that page, so a broken import or a typo'd export name surfaces as a real
  thrown exception instead of a silent no-op.
- If it's a **panel** (embeddable in the Web Terminal hub): support both
  standalone and embedded modes -- ``applyEmbedded()`` on load, an
  ``<osprey-theme-switcher>`` that hides itself when embedded (the hub owns
  theme chrome there), branding hidden when embedded -- plus a
  ``test_contract_params.py`` case. See :doc:`/how-to/embedding-osprey-panels`
  for the full dual-mode checklist and the well-known query/hash parameters.
- If the change is visible on screen: a ``test_visual.py`` ``TARGETS`` entry
  and a committed baseline PNG (regenerate with ``--regen-baselines`` on
  Linux/CI -- a baseline captured on macOS will mismatch there).
- If the change moves real behavior across a module boundary (a callback that
  used to call a sibling directly now goes through an injected factory, a
  delegator, or a re-exported method): an interaction pin that drives the
  whole chain through a real browser, not just each module's own
  mocked-neighbor Vitest suite.

**JSDoc / cast conventions** (``src/osprey/interfaces/vendor-globals.d.ts`` and
the exemplars below have worked examples):

- Vendored classic-script globals that never get real npm types (Plotly,
  ``marked``, ``hljs``, KaTeX) get **one** shared ambient-declarations file
  (``vendor-globals.d.ts``, ``declare const X: any;``) rather than a
  per-call-site cast.
- A ``document.getElementById``/``querySelector`` result that needs a
  property the generic ``Element``/``HTMLElement`` type doesn't have
  (``.value``, ``.checked``, ``.dataset``, ``.disabled``) gets an inline
  type-assertion cast to the concrete element type at the call site --
  ``/** @type {HTMLInputElement} */ (document.getElementById("foo"))``.
- For ``querySelectorAll(...).forEach(callback)`` where the callback needs a
  narrower element type than ``Element``: cast the **collection**, not the
  per-item callback parameter. TypeScript's contravariant
  function-parameter checking rejects a callback typed to accept only
  ``HTMLInputElement`` where one accepting any ``Element`` is expected, even
  though ``HTMLInputElement`` narrows ``Element``.
- ``catch (e)`` blocks that read the error message use
  ``e instanceof Error ? e.message : String(e)`` (``e`` is ``unknown`` under
  ``strict``); a ``catch`` that only logs the raw value needs no cast.
- Shared design-system helpers are imported by the absolute
  ``/design-system/js/*`` specifier, mapped in both ``tsconfig.json`` and
  ``vitest.config.js``'s ``resolve.alias`` -- the same import path resolves
  under the type-checker, Vitest, and the real browser.

**Exemplars** (concrete, complete, worth reading before writing your own):

- Vitest unit, factory-with-injected-callbacks pattern:
  ``tests/interfaces/artifacts/preview.test.mjs``,
  ``tests/interfaces/lattice_dashboard/render.test.mjs``,
  ``tests/interfaces/web_terminal/scaffold-detail.test.mjs``.
- Interaction pin, proving a multi-module split still wires up end to end:
  ``tests/interfaces/artifacts/test_gallery_interactions.py``,
  ``tests/interfaces/web_terminal/test_scaffold_detail.py``,
  ``tests/interfaces/lattice_dashboard/test_settings_form.py``,
  ``tests/interfaces/web_terminal/test_session_page.py``.
- Contract + dual-mode chrome: ``tests/interfaces/web_terminal/test_contract_params.py``.
- Visual baselines: ``tests/interfaces/design_system/test_visual.py``.
- Loads-clean: ``tests/interfaces/test_load_smokes.py``.

Docstrings
^^^^^^^^^^

All public functions, classes, and methods need Google-style docstrings:

.. code-block:: python

   def capability_function(param1: str, param2: int) -> bool:
       """Short description of function.

       Args:
           param1: Description of first parameter.
           param2: Description of second parameter.

       Returns:
           Description of return value.

       Raises:
           ValueError: When parameter is invalid.
       """

----

Community Guidelines
--------------------

**Code of Conduct**: We are committed to a welcoming and inclusive environment. Be respectful, welcome newcomers, accept constructive criticism, and show empathy. Harassment, personal attacks, trolling, or publishing private information are unacceptable. Report issues to the maintainers; all reports are handled confidentially.

**Communication Channels:**

- **GitHub Issues** -- Bug reports, feature requests, task tracking
- **GitHub Discussions** -- Questions, ideas, brainstorming
- **Pull Requests** -- Code contributions, documentation, code review

**Reporting Bugs**: Search existing issues first, then open a bug report with a clear description, reproduction steps, environment details (OS, Python version, Osprey version), and full error messages.

**Feature Requests**: Describe your use case, current limitations, proposed solution, and alternatives considered.

**Response Expectations**: Maintainers are volunteers. Please be patient and provide clear, detailed information.

Getting Help
------------

- `GitHub Discussions <https://github.com/als-apg/osprey/discussions>`_ -- Ask questions, share ideas
- `GitHub Issues <https://github.com/als-apg/osprey/issues>`_ -- Report bugs, request features
