Getting Started with Development
=================================

Set up your development environment and run your first tests.

Environment Setup
-----------------

**1. Fork and Clone**

.. code-block:: bash

   git clone https://github.com/YOUR-USERNAME/osprey.git
   cd osprey

**2. Create Virtual Environment**

.. code-block:: bash

   python3.11 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or: venv\Scripts\activate  # Windows

**3. Install Dependencies**

.. code-block:: bash

   pip install --upgrade pip
   pip install -e ".[dev,docs]"

**4. Verify Installation**

.. code-block:: bash

   pytest tests/ --ignore=tests/e2e -v

Running Tests
-------------

**Unit Tests (Fast, no API keys needed):**

.. code-block:: bash

   pytest tests/ --ignore=tests/e2e -v

**End-to-End Tests (Requires API keys):**

.. code-block:: bash

   cp env.example .env  # Edit to add your API keys
   pytest tests/e2e/ -v

Development Tools
-----------------

**Linting and Formatting (Ruff):**

.. code-block:: bash

   ruff check src/ tests/           # Check
   ruff check --fix src/ tests/     # Auto-fix
   ruff format src/ tests/          # Format

**Pre-Merge Checks:**

.. code-block:: bash

   ./scripts/premerge_check.sh

This scans for debug code, missing docstrings, and CHANGELOG entries.

**Build Documentation:**

.. code-block:: bash

   cd docs && make html

Common Issues
-------------

- **Import errors**: Run ``pip install -e ".[dev,docs]"``
- **E2E tests failing**: Check API keys in ``.env``
- **Doc build errors**: Run ``pip install -e ".[docs]"``

Next Steps
----------

- :doc:`02_git-and-github` - Git and GitHub workflow
- :doc:`03_code-standards` - Code style guidelines
- :doc:`04_developer-workflows` - Common workflows

