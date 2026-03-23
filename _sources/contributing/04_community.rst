Community Guidelines
====================

How to participate in the Osprey community.

.. tab-set::

   .. tab-item:: 📋 Community Guidelines

      **Code of Conduct**

      We are committed to a welcoming and inclusive environment.

      **Our Values:**

      - Be respectful and considerate
      - Welcome newcomers
      - Accept constructive criticism
      - Focus on what's best for the community
      - Show empathy

      **Unacceptable Behavior:**

      - Harassment or discrimination
      - Personal attacks
      - Trolling or inflammatory comments
      - Publishing private information
      - Other unwelcoming conduct

      **Reporting Issues:**

      If you experience unacceptable behavior, contact the maintainers. All reports are handled confidentially.

      ----

      **Communication Channels**

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - Channel
           - Use For
         * - **GitHub Issues**
           - Bug reports, feature requests, specific problems, task tracking
         * - **GitHub Discussions**
           - Questions, general discussions, ideas, brainstorming, announcements
         * - **Pull Requests**
           - Code contributions, documentation improvements, code review

      Follow :doc:`01_git-and-github` for the PR process.

      ----

      **Response Expectations**

      Maintainers are volunteers with day jobs and other commitments. Please:

      - Be patient while waiting for responses
      - Be respectful in all interactions
      - Provide clear, detailed information
      - Follow up constructively

      We aim to respond within a few days, but complex issues may take longer.

   .. tab-item:: 💬 I Need Help

      **Quick Start:**

      1. Check the documentation
      2. Search existing issues and discussions
      3. Still stuck? Ask for help

      **1. Check Documentation**

      - :doc:`index` - Environment setup and contribution guide
      - :doc:`03_ai-assisted-development` - AI workflows and tools
      - :doc:`../developer-guides/index` - Technical documentation

      **2. Search Existing Resources**

      Search GitHub Issues and Discussions - your question may already be answered.

      **3. Ask for Help**

      .. tab-set::

         .. tab-item:: Something's Not Working

            **Open a GitHub Issue**

            Include:

            - Clear description of the problem
            - What you've tried
            - Relevant code/configuration
            - Environment details (OS, Python version, Osprey version)

            **Be specific:**

            ❌ Bad: "Osprey doesn't work"

            ✅ Good: "Getting timeout when querying Channel Finder with wildcards"

         .. tab-item:: General Questions

            **Use GitHub Discussions**

            For:

            - Questions about using Osprey
            - General discussions and ideas
            - Brainstorming
            - Best practices

      **Help Others:**

      - Answer questions in issues and discussions
      - Review pull requests
      - Improve documentation

      .. note::
         Maintainers are volunteers. Please be patient and respectful while waiting for responses.

   .. tab-item:: 🐛 I Found a Bug

      **Before Reporting:**

      - Search existing issues - it may already be reported
      - Check if fixed in latest version
      - Verify it's actually a bug (not expected behavior)

      **Create a Bug Report**

      Include these details:

      1. **Clear description** - What happened vs. what you expected
      2. **Reproduction steps** - Minimal steps to reproduce the issue
      3. **Environment** - OS, Python version, Osprey version
      4. **Error messages** - Full stack traces
      5. **Additional context** - Any relevant details

      **Example Bug Report:**

      .. code-block:: text

         Title: Channel Finder timeout with wildcard queries

         Description: Queries fail with timeout when using broad wildcards.

         Steps to Reproduce:
         1. Run `osprey chat`
         2. Ask: "Find all PVs matching BPM:*:Current"
         3. Wait 30 seconds
         4. See timeout error

         Environment: macOS 14.1, Python 3.11.5, Osprey 0.9.7

         Error Message:
         ERROR: Channel Finder query timed out after 30.0 seconds

      **After Reporting:**

      - Maintainers will review your report
      - May ask follow-up questions for clarification
      - Will assign labels and priority
      - Will provide timeline if fix is planned

   .. tab-item:: 💡 I Have an Idea

      **Before Requesting:**

      - Search for similar feature requests
      - Check latest documentation - it might already exist
      - Consider if it fits Osprey's goals and scope

      **Create a Feature Request**

      Include these sections:

      1. **Use case** - What you're trying to accomplish
      2. **Current limitations** - What doesn't work today
      3. **Proposed solution** - Your ideal solution
      4. **Alternatives considered** - Other approaches you've thought about

      **Example Feature Request:**

      .. code-block:: text

         Title: Save and replay common queries

         Use Case:
         As a beamline scientist, I repeatedly ask similar questions.
         I want to save and replay them for faster workflow.

         Current Limitation:
         Must retype complex queries each time

         Proposed Solution:
         Add query saving functionality:
         > save query as "bpm_sector1"
         > run query "bpm_sector1"

         Benefits:
         - Faster workflow
         - Shareable queries between team members
         - Reproducible analysis

      **After Requesting:**

      - Community discussion on the idea
      - Maintainer review for feasibility
      - Priority assignment
      - Implementation (by maintainers or community members)

      **Want to Implement It Yourself?**

      Comment on the issue offering to help! Maintainers can provide guidance on:

      - Technical approach
      - Code structure
      - Testing requirements
      - Documentation needs

      See :doc:`01_git-and-github` for the contribution process.
