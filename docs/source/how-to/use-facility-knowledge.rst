.. _how-to-facility-knowledge:

==================
Facility Knowledge
==================

The Facility Knowledge system gives the OSPREY agent on-demand access to
structured narrative knowledge about your facility — subsystems, device
families, operational procedures, physics notes, and references.

It uses a two-altitude model, split by *when* the agent needs the content:

* **Always in context** — the facility's core operating context, carried in the
  Markdown rules under the project's ``.claude/rules/`` directory. These rule
  files load into the main agent's context at the start of every session, the
  same way ``CLAUDE.md`` does — the directory is just a convenient way to split
  instructions into separate topic files, not a different mechanism. Several of
  the rules are facility-specific: ``facility.md`` carries identity (name, type,
  mission), and others carry the control-system protocol, the facility timezone,
  and any safety or operating rules a build profile adds.
* **Fetched on demand** — the **Open Knowledge Format (OKF)** bundle, a
  directory of Markdown concept documents served by the
  ``osprey_facility_knowledge`` MCP server. This holds
  the facility's reference knowledge — subsystem descriptions, device specs,
  procedures, physics notes — and stays out of context until the agent (or you)
  retrieves it via ``list_concepts``, ``read_concept``, and ``search``.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the Open Knowledge Format is and where it comes from
   - The four core ideas: typed documents, progressive disclosure, cross-links, and validation levels
   - How to structure, configure, and author a facility knowledge bundle
   - The ``osprey knowledge`` CLI, the ``draft_concept`` tool, and the ``facility-knowledge`` subagent

   **Prerequisites:** The concepts need none. To try the mechanics you'll want
   a generated OSPREY project (the ``control_assistant`` preset ships a
   ready-made bundle).

The two tiers are documented separately:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Facility Rules
      :link: facility-rules
      :link-type: doc

      The always-in-context tier — what the ``control_assistant`` preset ships
      in ``.claude/rules/`` and how to change it.

   .. grid-item-card:: The OKF Bundle
      :link: okf-bundle
      :link-type: doc

      The on-demand tier — authoring, configuring, and serving the Open
      Knowledge Format bundle of concept documents.

.. toctree::
   :hidden:

   facility-rules
   okf-bundle
