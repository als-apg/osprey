Contributing to Osprey
======================

Thank you for your interest in contributing to the Osprey Framework!

Quick Start
-----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Getting Started
      :link: 01_getting-started
      :link-type: doc

      Set up your development environment and run your first tests.

   .. grid-item-card:: 🔄 Git & GitHub Workflow
      :link: 02_git-and-github
      :link-type: doc

      Learn branching conventions, commit messages, and the PR process.

   .. grid-item-card:: 📋 Code Standards
      :link: 03_code-standards
      :link-type: doc

      Python style guide, testing requirements, and linting setup.

   .. grid-item-card:: 🎯 Developer Workflows
      :link: 04_developer-workflows
      :link-type: doc

      Workflows for testing, commits, and documentation.

   .. grid-item-card:: 🤖 AI-Assisted Development
      :link: 05_ai-assisted-development
      :link-type: doc

      Use AI tools effectively with Osprey workflows.

   .. grid-item-card:: 🤝 Community Guidelines
      :link: 06_community
      :link-type: doc

      Code of conduct, reporting bugs, and getting help.

Before Your First Commit
-------------------------

**Development Requirements:**

- Python 3.11 or 3.12
- Virtual environment recommended
- Git for version control

**Pre-commit checklist:**

1. Quick check: ``./scripts/quick_check.sh`` (< 30 seconds)
2. Full CI check: ``./scripts/ci_check.sh`` (before push)
3. Pre-merge check: ``./scripts/premerge_check.sh`` (before PR)
4. Add CHANGELOG entry

See ``scripts/README.md`` for detailed script documentation.

**Every commit should:**

- Follow conventional commits format (``feat:``, ``fix:``, ``docs:``, etc.)
- Include CHANGELOG entry
- Have passing tests
- Include docstrings for new functions
- Be atomic (one logical change)

Using Workflow Files with AI
-----------------------------

Reference workflow files directly:

.. code-block:: text

   @docs/workflows/pre-merge-cleanup.md Scan my uncommitted changes
   @docs/workflows/commit-organization.md Help me organize my commits
   @docs/workflows/testing-workflow.md What type of test should I write?

See :doc:`04_developer-workflows` for complete workflow documentation.

Getting Help
------------

- `GitHub Issues <https://github.com/als-apg/osprey/issues>`_ - Report bugs, request features, get help when something isn't working
- `GitHub Discussions <https://github.com/als-apg/osprey/discussions>`_ - Ask questions, share ideas
- :doc:`../developer-guides/index` - Technical documentation

.. toctree::
   :maxdepth: 1
   :hidden:

   01_getting-started
   02_git-and-github
   03_code-standards
   04_developer-workflows
   05_ai-assisted-development
   06_community
