Contributing to Osprey
======================

Thank you for your interest in contributing to the Osprey Framework. This guide covers environment setup, Git workflow, code standards, the agent skills that automate them, and community guidelines.

----

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Development Setup
      :link: development-setup
      :link-type: doc
      :shadow: md

      Fork, install with ``uv``, wire up pre-commit — and the Python,
      front-end, and testing standards your change is held to.

   .. grid-item-card:: Git and GitHub Workflow
      :link: workflow
      :link-type: doc
      :shadow: md

      Branch naming, the three-tier local test run, conventional commits, and
      what happens to your pull request on the way to ``main``.

   .. grid-item-card:: Extending Osprey
      :link: extending-osprey
      :link-type: doc
      :shadow: md

      The seam map: connector, archiver, MCP server, chat bridge, ARIEL,
      health plugin, panel, LUME model -- each with the class or registry to
      subclass, a pinning test, and the how-to that configures it.

   .. grid-item-card:: Agent Skills
      :link: agent-skills
      :link-type: doc
      :shadow: md

      Six installable skills that teach a coding agent Osprey's own
      playbooks -- which one to install at each step of the contributor
      journey, and where the deployer-facing ones are documented.

----

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

.. toctree::
   :hidden:

   development-setup
   workflow
   extending-osprey
   agent-skills
