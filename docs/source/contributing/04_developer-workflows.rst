Developer Workflows
===================

Streamlined workflow guides for common development tasks, designed to work seamlessly Example assistants.

----

Workflow Categories
-------------------

.. tab-set::

   .. tab-item:: 🚀 Quick Workflows
      :sync: quick

      Fast workflows for common tasks (< 5 minutes).

      .. grid:: 2
         :gutter: 3

         .. grid-item-card:: 🔍 Pre-Merge Cleanup
            :link: #pre-merge-cleanup
            :link-type: ref

            Scan for common issues before committing.

            +++
            .. code-block:: bash

               ./scripts/premerge_check.sh

         .. grid-item-card:: 📝 Docstrings
            :link: #docstrings
            :link-type: ref

            Generate proper docstrings for functions and classes.

            +++
            .. code-block:: text

               @docs/workflows/docstrings.md

         .. grid-item-card:: 💬 Comments
            :link: #comments
            :link-type: ref

            Add strategic comments to complex code.

            +++
            .. code-block:: text

               @docs/workflows/comments.md

   .. tab-item:: 🏗️ Standard Workflows
      :sync: standard

      Comprehensive workflows for development tasks (10-30 minutes).

      .. grid:: 2
         :gutter: 3

         .. grid-item-card:: 🧪 Testing Strategy
            :link: #testing
            :link-type: ref

            Cost-aware testing: unit, integration, or e2e?

            +++
            .. code-block:: text

               @docs/workflows/testing-workflow.md

         .. grid-item-card:: 📦 Commit Organization
            :link: #commits
            :link-type: ref

            Organize changes into atomic commits with CHANGELOG entries.

            +++
            .. code-block:: text

               @docs/workflows/commit-organization.md

         .. grid-item-card:: 📚 Documentation Updates
            :link: #documentation
            :link-type: ref

            Identify and update documentation that needs changes.

            +++
            .. code-block:: text

               @docs/workflows/update-documentation.md

         .. grid-item-card:: 🤖 AI Code Review
            :link: #ai-review
            :link-type: ref

            Review AI-generated code for quality and correctness.

            +++
            .. code-block:: text

               @docs/workflows/ai-code-review.md

   .. tab-item:: 🎯 Release Workflows
      :sync: release

      Complete workflows for releases and major changes (1-2 hours).

      .. grid:: 1
         :gutter: 3

         .. grid-item-card:: 🚢 Release Process
            :link: #release
            :link-type: ref

            Complete release workflow with testing, versioning, and deployment.

            +++
            .. code-block:: text

               @docs/workflows/release-workflow.md Guide me through releasing v0.9.8

----

Detailed Workflow Guides
------------------------

.. _pre-merge-cleanup:

🔍 Pre-Merge Cleanup
^^^^^^^^^^^^^^^^^^^^

.. workflow-summary:: ../../workflows/pre-merge-cleanup.md
   :show-use-when:

**Command Line:**

.. code-block:: bash

   ./scripts/premerge_check.sh

**Example:**

.. code-block:: text

   @docs/workflows/pre-merge-cleanup.md Scan my uncommitted changes

**What it checks:**

- Debug code (``print()``, ``breakpoint()``, etc.)
- Missing or incomplete docstrings
- TODO/FIXME comments
- Missing CHANGELOG entries
- Import organization

----

.. _commits:

📦 Commit Organization
^^^^^^^^^^^^^^^^^^^^^^

.. workflow-summary:: ../../workflows/commit-organization.md
   :show-use-when:

**Example:**

.. code-block:: text

   @docs/workflows/commit-organization.md Help me organize my commits

**Best for:**

- Feature branches with multiple related changes
- Refactoring efforts spanning multiple files
- Bug fixes that touch multiple components
- First-time contributors organizing their PR

.. note::
   Each commit gets its own CHANGELOG entry. Don't batch all entries at the start!

----

.. _testing:

🧪 Testing Strategy
^^^^^^^^^^^^^^^^^^^

.. workflow-summary:: ../../workflows/testing-workflow.md
   :show-use-when:

**Example:**

.. code-block:: text

   @docs/workflows/testing-workflow.md What type of test should I write?

**Decision Framework:**

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Test Type
     - When to Use
     - Cost/Speed
   * - **Unit**
     - Pure functions, business logic, utilities
     - ⚡ Fast, cheap
   * - **Integration**
     - Component interactions, API endpoints
     - ⚙️ Medium speed/cost
   * - **E2E**
     - Critical user flows, deployment validation
     - 🐌 Slow, expensive

----

.. _documentation:

📚 Documentation Updates
^^^^^^^^^^^^^^^^^^^^^^^^

.. workflow-summary:: ../../workflows/update-documentation.md
   :show-use-when:

**Example:**

.. code-block:: text

   @docs/workflows/update-documentation.md What docs need updating?

----

.. _docstrings:

📝 Docstrings
^^^^^^^^^^^^^

.. workflow-summary:: ../../workflows/docstrings.md
   :show-use-when:

**Example:**

.. code-block:: text

   @docs/workflows/docstrings.md Write a docstring for this function

----

.. _comments:

💬 Comments
^^^^^^^^^^^

.. workflow-summary:: ../../workflows/comments.md
   :show-use-when:

**Example:**

.. code-block:: text

   @docs/workflows/comments.md Add comments to explain this logic

----

.. _ai-review:

🤖 AI Code Review
^^^^^^^^^^^^^^^^^

.. workflow-summary:: ../../workflows/ai-code-review.md
   :show-use-when:

**Example:**

.. code-block:: text

   @docs/workflows/ai-code-review.md Review this AI-generated code

----

.. _release:

🚢 Release Workflow
^^^^^^^^^^^^^^^^^^^

.. workflow-summary:: ../../workflows/release-workflow.md
   :show-use-when:

**Example:**

.. code-block:: text

   @docs/workflows/release-workflow.md Guide me through releasing v0.9.8

----

Common Workflow Patterns
-------------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🎯 Before Every Commit

      .. code-block:: text

         @docs/workflows/pre-merge-cleanup.md Scan for issues

      Catches common mistakes automatically.

   .. grid-item-card:: 🎨 First PR?

      .. code-block:: text

         @docs/workflows/commit-organization.md + @CONTRIBUTING.md
         Review my changes against contribution guidelines

      Get your PR ready for review.

   .. grid-item-card:: ✨ Adding Features?

      .. code-block:: text

         @docs/workflows/update-documentation.md
         What documentation needs updating?

      Don't forget the docs!

   .. grid-item-card:: 🚀 Ready to Release?

      .. code-block:: text

         @docs/workflows/release-workflow.md Guide me through the release

      Follow the complete release checklist.

----

Next Steps
----------

.. seealso::

   **Explore More:**

   - Browse detailed workflow files in ``docs/workflows/``
   - :doc:`05_ai-assisted-development` for AI integration patterns
   - :doc:`03_code-standards` for coding conventions
   - :doc:`index` for contributing overview

   **Get Started:**

   Pick a workflow card above and try it with your next change!
