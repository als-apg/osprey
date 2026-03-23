AI-Assisted Development
========================

Osprey's workflows are designed for AI coding assistants. Use structured prompts to automate testing, documentation, commits, and releases.

----

Getting Started
---------------

Osprey workflows work with AI coding assistants. Choose the tool that fits your workflow:

.. tab-set::

   .. tab-item:: 🎯 Cursor

      **AI-powered code editor** with native ``@`` mentions for workflow files.

      **Platform Support:** Windows, macOS, Linux

      .. dropdown:: Installation Instructions
         :color: info
         :icon: download

         **1. Download:**

         Visit `cursor.com <https://cursor.com>`_ and download for your platform.

         **2. Install:**

         - **Windows:** Run the ``.exe`` installer
         - **macOS:** Open the ``.dmg`` and drag to Applications
         - **Linux:** Install ``.deb`` or ``.rpm`` package for your distribution

         **3. First-Time Setup:**

         - Launch Cursor
         - Complete the setup wizard
         - Create an account to unlock AI features (optional but recommended)
         - Open the Osprey project folder

         .. seealso::
            Official docs: `docs.cursor.com <https://docs.cursor.com/en/get-started/installation>`_

   .. tab-item:: 💬 Claude Code

      **Terminal-based AI assistant** that works alongside your existing editor.

      **Platform Support:** Windows (WSL), macOS, Linux

      .. dropdown:: Installation Instructions
         :color: info
         :icon: download

         **1. Prerequisites:**

         - Node.js 18.0 or higher
         - Internet connection

         **2. Install:**

         .. code-block:: bash

            # Using npm (recommended)
            npm install -g @anthropic-ai/claude-code

            # Or using Homebrew (macOS)
            brew install anthropic/tap/claude-code

         **3. Verify Installation:**

         .. code-block:: bash

            claude --version

         **4. Authenticate:**

         .. code-block:: bash

            claude auth login

         This opens your browser to sign in with your Anthropic account.

         **5. Update:**

         .. code-block:: bash

            # Using npm
            npm update -g @anthropic-ai/claude-code

            # Using Homebrew
            brew upgrade claude-code

         .. seealso::
            Official docs: `docs.anthropic.com <https://docs.anthropic.com/en/docs/claude-code/getting-started>`_

----

Accessing Workflow Files
------------------------

Osprey's AI workflow files are bundled with the installed package. Choose the method that fits your AI tool:

**Option 1: Copy to Project (Any AI Tool)**

Copy tasks to your project's ``.ai-tasks/`` directory for use with any AI assistant:

.. code-block:: bash

   # Browse tasks interactively (recommended)
   osprey tasks

   # Or copy directly by name
   osprey tasks copy pre-merge-cleanup

Then reference in your AI assistant:

.. code-block:: text

   @.ai-tasks/pre-merge-cleanup/instructions.md Scan my changes

**Option 2: Install as Claude Code Skills**

For Claude Code users, install tasks as skills for automatic discovery:

.. code-block:: bash

   # Install a task as a Claude Code skill
   osprey claude install migrate

Then simply ask Claude to use the skill - it discovers installed skills automatically.

.. dropdown:: Command Reference and Advanced Usage
   :color: info
   :icon: terminal

   **Task Commands** (browse and copy tasks)

   .. code-block:: bash

      # Interactive browser (recommended)
      osprey tasks

      # List all available tasks
      osprey tasks list

      # Copy task to project (.ai-tasks/)
      osprey tasks copy pre-commit

      # View task instructions
      osprey tasks show pre-commit

      # Get path to task file
      osprey tasks path pre-commit

   **Claude Commands** (manage Claude Code skills)

   .. code-block:: bash

      # Install task as Claude Code skill
      osprey claude install migrate

      # Install with force overwrite
      osprey claude install migrate --force

      # List installed and available skills
      osprey claude list

   **Viewing Available Tasks**

   List all tasks bundled with Osprey:

   .. code-block:: bash

      osprey tasks list

   This will show tasks like:

   .. code-block:: text

      migrate                          Upgrade downstream projects to newer OSPREY versions
      pre-commit                       Validate code before commits
      testing-workflow                 Comprehensive testing guide
      commit-organization              Organize changes into atomic commits
      pre-merge-cleanup                Detect loose ends before merging
      ai-code-review                   Review AI-generated code
      docstrings                       Write Sphinx-compatible docstrings
      comments                         Write purposeful inline comments
      release-workflow                 Create releases with proper versioning
      update-documentation             Keep docs in sync with code changes
      create-capability                Create new capabilities in Osprey apps
      channel-finder-pipeline-selection   Select appropriate Channel Finder pipelines
      channel-finder-database-builder  Build channel finder databases

   **Skill Auto-Generation**

   Tasks with ``skill_description`` in their frontmatter can be installed as
   Claude Code skills without requiring custom wrappers:

   .. code-block:: bash

      # List installable skills (shows auto-generated)
      osprey claude list

      # Install any skill-ready task
      osprey claude install create-capability

   **Interactive Menu**

   You can also access tasks from the interactive menu:

   .. code-block:: bash

      # Launch interactive menu
      osprey

      # Select: [>] tasks - Browse AI assistant tasks
      # Or: [>] claude - Manage Claude Code skills

   **Version Updates**

   Task files are version-locked with your installed Osprey version. After upgrading Osprey, reinstall tasks to get updates:

   .. code-block:: bash

      # After upgrading Osprey
      pip install --upgrade osprey-framework

      # Reinstall tasks
      osprey claude install migrate --force

----

Workflow Catalog
----------------

.. tab-set::

   .. tab-item:: 🚀 Quick Workflows
      :sync: quick

      Fast workflows for common tasks (< 5 minutes).

      .. grid:: 2
         :gutter: 3

         .. grid-item-card:: 🔍 Pre-Merge Cleanup
            :link: pre-merge-cleanup
            :link-type: ref

            Scan for common issues before committing.

         .. grid-item-card:: 📝 Docstrings
            :link: docstrings
            :link-type: ref

            Generate proper docstrings for functions and classes.

         .. grid-item-card:: 💬 Comments
            :link: comments
            :link-type: ref

            Add strategic comments to complex code.

   .. tab-item:: 🏗️ Standard Workflows
      :sync: standard

      Comprehensive workflows for development tasks (10-30 minutes).

      .. grid:: 2
         :gutter: 3

         .. grid-item-card:: 🧪 Testing Strategy
            :link: testing
            :link-type: ref

            Cost-aware testing: unit, integration, or e2e?

         .. grid-item-card:: 📦 Commit Organization
            :link: commits
            :link-type: ref

            Organize changes into atomic commits with CHANGELOG entries.

         .. grid-item-card:: 📚 Documentation Updates
            :link: documentation
            :link-type: ref

            Identify and update documentation that needs changes.

         .. grid-item-card:: 🤖 AI Code Review
            :link: ai-review
            :link-type: ref

            Review AI-generated code for quality and correctness.

         .. grid-item-card:: 🔧 Create Capability
            :link: create-capability
            :link-type: ref

            Build new capabilities for Osprey applications.

         .. grid-item-card:: 🔍 Channel Finder Pipeline Selection
            :link: channel-finder-pipeline
            :link-type: ref

            Choose the right pipeline for your control system.

         .. grid-item-card:: 🗄️ Channel Finder Database Builder
            :link: channel-finder-database
            :link-type: ref

            Build high-quality channel databases with AI assistance.

   .. tab-item:: 🎯 Release Workflows
      :sync: release

      Complete workflows for releases and major changes (1-2 hours).

      .. grid:: 1
         :gutter: 3

         .. grid-item-card:: 🚢 Release Process
            :link: release
            :link-type: ref

            Complete release workflow with testing, versioning, and deployment.

----

Detailed Workflow Guides
------------------------

.. tip::
   **First-time setup:** Before using any workflow, copy it to your project:

   .. code-block:: bash

      osprey tasks copy pre-merge-cleanup

   Then use ``@.ai-tasks/<task>/instructions.md`` in your AI assistant.

.. _pre-merge-cleanup:

🔍 Pre-Merge Cleanup
^^^^^^^^^^^^^^^^^^^^

`View task file <https://github.com/als-apg/osprey/blob/main/src/osprey/assist/tasks/pre-merge-cleanup/instructions.md>`_

**Command Line:**

.. code-block:: bash

   ./scripts/premerge_check.sh

**Example:**

.. code-block:: text

   @.ai-tasks/pre-merge-cleanup/instructions.md Scan my uncommitted changes

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

`View task file <https://github.com/als-apg/osprey/blob/main/src/osprey/assist/tasks/commit-organization/instructions.md>`_

**Example:**

.. code-block:: text

   @.ai-tasks/commit-organization/instructions.md Help me organize my commits

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

`View task file <https://github.com/als-apg/osprey/blob/main/src/osprey/assist/tasks/testing-workflow/instructions.md>`_

**Example:**

.. code-block:: text

   @.ai-tasks/testing-workflow/instructions.md What type of test should I write?

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

`View task file <https://github.com/als-apg/osprey/blob/main/src/osprey/assist/tasks/update-documentation/instructions.md>`_

**Example:**

.. code-block:: text

   @.ai-tasks/update-documentation/instructions.md What docs need updating?

----

.. _docstrings:

📝 Docstrings
^^^^^^^^^^^^^

`View task file <https://github.com/als-apg/osprey/blob/main/src/osprey/assist/tasks/docstrings/instructions.md>`_

**Example:**

.. code-block:: text

   @.ai-tasks/docstrings/instructions.md Write a docstring for this function

----

.. _comments:

💬 Comments
^^^^^^^^^^^

`View task file <https://github.com/als-apg/osprey/blob/main/src/osprey/assist/tasks/comments/instructions.md>`_

**Example:**

.. code-block:: text

   @.ai-tasks/comments/instructions.md Add comments to explain this logic

----

.. _ai-review:

🤖 AI Code Review
^^^^^^^^^^^^^^^^^

`View task file <https://github.com/als-apg/osprey/blob/main/src/osprey/assist/tasks/ai-code-review/instructions.md>`_

**Example:**

.. code-block:: text

   @.ai-tasks/ai-code-review/instructions.md Review this AI-generated code

----

.. _create-capability:

🔧 Create Capability
^^^^^^^^^^^^^^^^^^^^

`View task file <https://github.com/als-apg/osprey/blob/main/src/osprey/assist/tasks/create-capability/instructions.md>`_

**Install as Claude Code skill:**

.. code-block:: bash

   osprey claude install create-capability

**Example:**

.. code-block:: text

   @.ai-tasks/create-capability/instructions.md Help me create a new capability

**What it covers:**

- Requirements gathering (inputs, outputs, dependencies)
- Context class design (data structures)
- Capability implementation (business logic)
- Registry configuration
- Testing patterns

**Interactive workflow:** The task guides you through questions before writing code,
ensuring proper design of context classes, error handling, and registry entries.

----

.. _channel-finder-pipeline:

🔍 Channel Finder Pipeline Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View task file <https://github.com/als-apg/osprey/blob/main/src/osprey/assist/tasks/channel-finder-pipeline-selection/instructions.md>`_

**Example:**

.. code-block:: text

   @.ai-tasks/channel-finder-pipeline-selection/instructions.md Help me select the right Channel Finder pipeline.

----

.. _channel-finder-database:

🗄️ Channel Finder Database Builder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`View task file <https://github.com/als-apg/osprey/blob/main/src/osprey/assist/tasks/channel-finder-database-builder/instructions.md>`_

**Example:**

.. code-block:: text

   @.ai-tasks/channel-finder-database-builder/instructions.md Help me build my Channel Finder database.


----

.. _release:

🚢 Release Workflow
^^^^^^^^^^^^^^^^^^^

`View task file <https://github.com/als-apg/osprey/blob/main/src/osprey/assist/tasks/release-workflow/instructions.md>`_

**Example:**

.. code-block:: text

   @.ai-tasks/release-workflow/instructions.md Guide me through releasing v0.9.8

----

Best Practices
--------------

**Do:**

- Reference specific workflows with ``@`` mentions
- Provide context about what you're working on
- Review all AI-generated code carefully
- Run tests to verify AI changes
- Check for security issues in AI code

**Don't:**

- Blindly accept AI suggestions
- Skip testing AI-generated code
- Assume AI knows project-specific details
- Skip pre-merge cleanup checks
- Use AI to write every single line
- Skip human code review

----

Example: Adding a Capability with AI
-------------------------------------

**1. Use the create-capability workflow:**

.. code-block:: text

   @.ai-tasks/create-capability/instructions.md
   Help me create a new capability for archiver data retrieval.

Or install as a Claude Code skill for automatic discovery:

.. code-block:: bash

   osprey claude install create-capability

Then simply ask Claude: *"Help me create a new capability for archiver data"*

The workflow will guide you through:

- Requirements gathering (what data, inputs, outputs)
- Context class design
- Capability implementation
- Registry configuration

**2. Add appropriate tests:**

.. code-block:: text

   @.ai-tasks/testing-workflow/instructions.md
   My capability calls an external API. Should I write unit or integration tests?

**3. Update documentation:**

.. code-block:: text

   @.ai-tasks/update-documentation/instructions.md
   I added a new archiver capability. What documentation needs updating?

**4. Pre-commit cleanup:**

.. code-block:: text

   @.ai-tasks/pre-merge-cleanup/instructions.md
   Scan my uncommitted changes for issues

**5. Organize commits:**

.. code-block:: text

   @.ai-tasks/commit-organization/instructions.md
   Help me organize these changes into atomic commits with CHANGELOG entries

----

.. seealso::

   **Explore More:**

   - List available tasks: ``osprey tasks list``
   - :doc:`02_code-standards` for coding conventions
   - :doc:`index` for environment setup
   - :doc:`../developer-guides/index` for technical guides

   **Get Started:**

   Run ``osprey tasks list`` to see available tasks, then reference one with your next change!
