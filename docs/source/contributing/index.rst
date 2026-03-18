Contributing to Osprey
======================

Thank you for your interest in contributing to the Osprey Framework. This guide covers environment setup, Git workflow, code standards, AI-assisted development, and community guidelines.

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

   # Tier 3: Pre-merge check -- before creating a PR
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
3. PR requirements: pass all CI checks, receive maintainer approval, include CHANGELOG entries, have appropriate tests.
4. During review: respond to feedback promptly, make requested changes, ask questions if unclear.

----

Code Standards
--------------

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

   # With coverage
   uv run pytest tests/ --ignore=tests/e2e --cov=src/osprey

.. warning::

   E2E tests **must** be run with ``pytest tests/e2e/`` not ``pytest -m e2e``.
   The marker-based approach causes registry state leaks and service conflicts.

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

AI-Assisted Development
-----------------------

Osprey includes structured AI workflow tasks for coding assistants (Claude Code, Cursor, etc.).

Accessing Workflow Tasks
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Browse tasks interactively
   osprey tasks

   # List all available tasks
   osprey tasks list

   # Copy a task to your project
   osprey tasks copy pre-merge-cleanup

   # Install as a Claude Code skill
   osprey claude install migrate

Available tasks include: ``pre-merge-cleanup``, ``testing-workflow``, ``commit-organization``,
``docstrings``, ``comments``, ``ai-code-review``, ``create-capability``, ``release-workflow``,
``update-documentation``, ``channel-finder-pipeline-selection``, ``channel-finder-database-builder``,
and ``migrate``.

Using Tasks
^^^^^^^^^^^

**With any AI assistant** -- reference the copied task file:

.. code-block:: text

   @.ai-tasks/pre-merge-cleanup/instructions.md Scan my uncommitted changes

**With Claude Code** -- install as a skill for automatic discovery:

.. code-block:: bash

   osprey claude install create-capability

Then ask Claude directly: *"Help me create a new capability for archiver data retrieval."*

Best Practices
^^^^^^^^^^^^^^

- Reference specific workflows with ``@`` mentions and provide context.
- Always review AI-generated code carefully.
- Run tests to verify AI changes.
- Check for security issues in AI-generated code.
- Do not blindly accept suggestions or skip pre-merge cleanup.

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
- ``osprey tasks list`` -- Browse available AI workflow tasks
