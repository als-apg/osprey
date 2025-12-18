Community Guidelines
====================

How to participate in the Osprey community.

Code of Conduct
---------------

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
- Trolling
- Publishing private information
- Unwelcoming conduct

**Reporting:**

If you experience unacceptable behavior, contact maintainers. All reports are confidential.

Reporting Bugs
--------------

**Before reporting:**

- Search existing issues
- Check if fixed in latest version
- Verify it's actually a bug

**Create a bug report with:**

1. **Clear description** - What happened vs. expected
2. **Reproduction steps** - Minimal steps to reproduce
3. **Environment** - OS, Python version, Osprey version
4. **Error messages** - Full stack traces
5. **Additional context** - Any relevant details

**Example:**

.. code-block:: text

   Title: Channel Finder timeout with wildcard queries

   Description: Queries fail with timeout when using broad wildcards.

   Steps:
   1. `osprey chat`
   2. Ask: "Find all PVs matching BPM:*:Current"
   3. Wait 30 seconds
   4. See timeout error

   Environment: macOS 14.1, Python 3.11.5, Osprey 0.9.7

   Error:
   ERROR: Channel Finder query timed out after 30.0 seconds

**After reporting:**

- Maintainers will review your report
- May ask follow-up questions
- Will assign labels and priority
- Will provide timeline if fix is planned

Requesting Features
-------------------

**Before requesting:**

- Search for similar requests
- Check latest documentation
- Consider if it fits Osprey's goals

**Create a feature request with:**

1. **Use case** - What you're trying to accomplish
2. **Current limitations** - What doesn't work today
3. **Proposed solution** - Your ideal solution
4. **Alternatives considered** - Other approaches

**Example:**

.. code-block:: text

   Title: Save and replay common queries

   Use Case: As a beamline scientist, I repeatedly ask similar
   questions. I want to save and replay them.

   Limitation: Must retype complex queries each time

   Solution: Add query saving:
   > save query as "bpm_sector1"
   > run query "bpm_sector1"

   Benefits: Faster workflow, shareable queries

**After requesting:**

- Community discussion
- Maintainer review
- Priority assignment
- Implementation (by maintainers or community)

**Want to implement it yourself?**

Comment on the issue offering to help, get guidance from maintainers.

Communication Channels
----------------------

**GitHub Issues**

Use GitHub Issues for:

- Bug reports
- Feature requests
- When something isn't working
- When you need help with a specific problem
- Tracking tasks

**GitHub Discussions**

Use GitHub Discussions for:

- Questions about using Osprey
- General discussions and ideas
- Brainstorming
- Announcements

**Pull Requests**

Use Pull Requests for:

- Code contributions
- Documentation improvements
- Code review

Follow :doc:`02_git-and-github` for PR process.

**Note:** Maintainers are volunteers. Please be patient and respectful while waiting for responses.

Getting Help
------------

**1. Check documentation:**

- :doc:`01_getting-started` - Setup
- :doc:`04_developer-workflows` - Workflows
- Developer guides - Technical docs

**2. Search existing issues and discussions**

**3. Something not working? Open a GitHub Issue**

Include:

- Clear description of the problem
- What you've tried
- Relevant code/configuration
- Environment details

**Be specific:**

Bad: "Osprey doesn't work"
Good: "Getting timeout when querying Channel Finder with wildcards"

**4. General questions? Use GitHub Discussions**

For questions about usage, ideas, or general discussion.

**Help others:**

- Answer questions in issues and discussions
- Review pull requests
- Improve documentation

Recognition
-----------

We value all contributions:

- First contributions are celebrated
- Contributors listed in release notes
- Significant contributions get special recognition
- All help appreciated

Thank you for being part of the Osprey community!

Next Steps
----------

- :doc:`01_getting-started` - Set up your environment
- :doc:`02_git-and-github` - Learn the workflow
- :doc:`05_ai-assisted-development` - Use AI tools
- Find "good first issue" labels on GitHub
