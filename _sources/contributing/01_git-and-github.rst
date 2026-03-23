Git and GitHub Workflow
=======================

Git and GitHub workflow for contributing to Osprey.

Branch Naming
-------------

- ``feature/description`` - New features
- ``fix/description`` - Bug fixes
- ``docs/description`` - Documentation
- ``refactor/description`` - Code refactoring
- ``test/description`` - Test improvements

Making Changes
--------------

**1. Create Branch:**

.. code-block:: bash

   git checkout -b feature/your-feature-name

**2. Make Changes:**

- Follow :doc:`03_code-standards`
- Add tests
- Update documentation
- Run linters

**3. Test Changes:**

Run tests in three tiers:

.. code-block:: bash

   # Tier 1: Quick check (< 30 seconds) - Run before every commit
   ./scripts/quick_check.sh

   # Tier 2: Full CI check (2-3 minutes) - Run before pushing
   ./scripts/ci_check.sh

   # Tier 3: Pre-merge check - Run before creating PR
   ./scripts/premerge_check.sh

**Why test locally?**

- ``quick_check.sh``: Catches 90% of issues instantly
- ``ci_check.sh``: Replicates exact GitHub Actions checks, saves CI minutes
- ``premerge_check.sh``: Comprehensive validation against target branch

**4. Commit Changes:**

.. code-block:: bash

   git add .
   git commit -m "feat: Add comprehensive contributing guide

   - Document environment setup
   - Add workflow guidelines"

.. tip::

   Install pre-commit hooks to run checks automatically on every commit:

   .. code-block:: bash

      pre-commit install

   Hooks will auto-fix formatting issues and prevent commits with common problems.

Commit Message Format
---------------------

Follow conventional commits:

- ``feat:`` - New features
- ``fix:`` - Bug fixes
- ``docs:`` - Documentation
- ``refactor:`` - Code refactoring
- ``test:`` - Tests
- ``chore:`` - Dependencies, build

**Good:**

.. code-block:: text

   feat: Add Read the Docs integration

   - Create .readthedocs.yaml configuration
   - Configure docs dependencies in pyproject.toml

**Bad:**

.. code-block:: text

   update stuff
   WIP

CHANGELOG Entries
-----------------

Every commit needs a CHANGELOG entry. Add it **before** committing:

.. code-block:: markdown

   ## [Unreleased]

   ### Added
   - Developer workflow guides

   ### Fixed
   - Channel Finder timeout issue

.. note::

   For multiple commits, add CHANGELOG entries one at a time for each commit.

Pull Request Process
--------------------

**1. Push Branch:**

.. code-block:: bash

   git push origin feature/your-feature-name

**2. Open PR:**

- Go to GitHub repository
- Click "New Pull Request"
- Fill out PR template with:

  - Description of changes
  - Related issues
  - Testing performed

**3. PR Requirements:**

- Pass all CI checks
- Receive maintainer approval
- Include CHANGELOG entries
- Have appropriate tests

Code Review
-----------

**Before requesting review:**

- Run pre-merge cleanup
- Verify tests pass
- Update documentation

**During review:**

- Respond to feedback promptly
- Make requested changes
- Ask questions if unclear

Testing Scripts
---------------

**Local CI Replication:**

The ``ci_check.sh`` script runs the exact same checks as GitHub Actions:

1. **Linting**: Ruff linting, formatting, and mypy type checking
2. **Tests**: Full pytest suite with coverage reporting
3. **Documentation**: Sphinx build and link checking
4. **Package**: Build and validate package with twine

If ``ci_check.sh`` passes locally, GitHub Actions CI will almost certainly pass.

**See**: ``scripts/README.md`` for complete documentation of all testing scripts.

Next Steps
----------

- :doc:`02_code-standards` - Code style guidelines
- :doc:`03_ai-assisted-development` - AI workflows
