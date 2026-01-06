Contributing to Osprey
======================

Thank you for your interest in contributing to the Osprey Framework!

This guide will help you get set up and ready to contribute. Whether you're fixing a bug, adding a feature, or improving documentation, we're excited to have you here.

----

Contributing Guide
------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🔄 Git & GitHub Workflow
      :link: 01_git-and-github
      :link-type: doc

      Learn branching conventions, commit messages, and the PR process.

   .. grid-item-card:: 📋 Code Standards
      :link: 02_code-standards
      :link-type: doc

      Python style guide, testing requirements, and linting setup.

   .. grid-item-card:: 🤖 AI-Assisted Development
      :link: 03_ai-assisted-development
      :link-type: doc

      Structured workflows designed for AI coding assistants.

   .. grid-item-card:: 🤝 Community Guidelines
      :link: 04_community
      :link-type: doc

      Code of conduct, reporting bugs, and getting help.

----

Environment Setup
-----------------

Before you can contribute, you'll need to set up your local development environment. This process takes about 5-10 minutes for first-time setup.

**Prerequisites:**

- Python 3.11 or 3.12
- Git for version control
- A GitHub account

**1. Fork and Clone**

First, fork the Osprey repository on GitHub, then clone your fork locally:

.. code-block:: bash

   git clone https://github.com/YOUR-USERNAME/osprey.git
   cd osprey

**2. Create Virtual Environment**

We strongly recommend using a virtual environment to isolate dependencies:

.. code-block:: bash

   python3.11 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or: venv\Scripts\activate  # Windows

**3. Install Dependencies**

Install Osprey in development mode with all development and documentation dependencies:

.. code-block:: bash

   pip install --upgrade pip
   pip install -e ".[dev,docs]"

This installs the framework in "editable" mode, meaning changes you make to the source code are immediately available.

**4. Verify Installation**

Run the test suite to make sure everything is working:

.. code-block:: bash

   pytest tests/ --ignore=tests/e2e -v

If all tests pass, you're ready to start contributing!

----

Quick Reference
---------------

**Common Commands:**

.. code-block:: bash

   # Run tests
   pytest tests/ --ignore=tests/e2e -v

   # Check code style
   ruff check src/ tests/

   # Pre-commit check
   ./scripts/premerge_check.sh

   # Build documentation
   cd docs && sphinx-autobuild source build

**Before Every Commit:**

- Run ``./scripts/premerge_check.sh`` to catch common issues
- Follow conventional commit format (``feat:``, ``fix:``, ``docs:``, etc.)
- Add a CHANGELOG entry
- Ensure tests pass

See the guide cards above for detailed workflows and standards.

----

Getting Help
------------

**Stuck? Have questions?**

- `GitHub Discussions <https://github.com/als-apg/osprey/discussions>`_ - Ask questions, share ideas, get help from the community
- `GitHub Issues <https://github.com/als-apg/osprey/issues>`_ - Report bugs, request features
- :doc:`../developer-guides/index` - Deep technical documentation on framework architecture
- :doc:`04_community` - Community guidelines and code of conduct



.. toctree::
   :maxdepth: 1
   :hidden:

   01_git-and-github
   02_code-standards
   03_ai-assisted-development
   04_community
