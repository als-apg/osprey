AI-Assisted Development
========================

How to use AI coding assistants effectively with Osprey.

Overview
--------

Osprey embraces AI-assisted development with structured workflows and prompts designed for AI tools.

**Benefits:**

- Faster onboarding
- Consistent quality
- Automated checklists
- Pre-written prompts

Why It Works
------------

Osprey provides:

- YAML frontmatter in workflows for machine parsing
- Step-by-step checklists
- Copy-paste ready prompts
- Code examples and patterns

Recommended Tools
-----------------

**Cursor** (Recommended)

- Built on VS Code
- Native @ mentions
- Excellent context management

**GitHub Copilot**

- Works in many editors
- Good code completion
- Chat interface

Basic Usage
-----------

Reference workflow files in your prompts:

.. code-block:: text

   @docs/workflows/pre-merge-cleanup.md Scan my uncommitted changes

The AI will:

1. Read the workflow file
2. Analyze your changes
3. Check against checklists
4. Report issues with locations
5. Generate action plan

Common Workflows
----------------

**Before Committing:**

.. code-block:: text

   @docs/workflows/pre-merge-cleanup.md Scan my uncommitted changes
   @docs/workflows/commit-organization.md Help me organize my commits

**While Writing Code:**

.. code-block:: text

   @docs/workflows/docstrings.md Write a docstring for this function
   @docs/workflows/testing-workflow.md What type of test should I write?

**After Changes:**

.. code-block:: text

   @docs/workflows/update-documentation.md What docs need updating?

**Before Release:**

.. code-block:: text

   @docs/workflows/release-workflow.md Guide me through releasing v0.9.8

Best Practices
--------------

**Do:**

- Reference specific workflows
- Be specific in prompts
- Provide context
- Review all AI output
- Run tests to verify
- Check for security issues

**Don't:**

- Blindly accept code
- Skip testing AI code
- Assume AI knows project details
- Skip pre-merge checks
- Use AI for every line
- Skip code review

Example: Adding a Capability
----------------------------

**1. Plan:**

.. code-block:: text

   @docs/workflows/ + @docs/source/developer-guides/
   I want to add a capability for historical archiver data. Help me plan.

**2. Write Code:**

.. code-block:: text

   @docs/workflows/docstrings.md
   Write a docstring for my new capability

**3. Add Tests:**

.. code-block:: text

   @docs/workflows/testing-workflow.md
   My capability calls an external API. Should I write unit or integration tests?

**4. Update Docs:**

.. code-block:: text

   @docs/workflows/update-documentation.md
   I added a new capability. What documentation needs updating?

**5. Pre-Commit:**

.. code-block:: text

   @docs/workflows/pre-merge-cleanup.md
   Scan my uncommitted changes

**6. Organize Commits:**

.. code-block:: text

   @docs/workflows/commit-organization.md
   Help me organize these changes into atomic commits

Common Issues
-------------

**AI gives generic responses:**

.. code-block:: text

   Please follow patterns in @docs/workflows/docstrings.md for Osprey

**AI misses context:**

.. code-block:: text

   @docs/workflows/ + @docs/source/developer-guides/01_overview/
   [your question with context]

**AI suggests non-standard approach:**

.. code-block:: text

   Does that follow @docs/workflows/testing-workflow.md?
   Our project prefers unit tests with mocking for external APIs.


See :doc:`04_developer-workflows` for complete workflow documentation.
