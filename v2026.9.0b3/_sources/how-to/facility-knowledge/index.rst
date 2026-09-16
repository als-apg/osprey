.. _how-to-facility-knowledge:

==================
Facility Knowledge
==================

For the OSPREY agent to be useful at your facility, it has to know your
facility: what the subsystems are, how devices are named, which procedures
apply, and where any of it sits on the control system. OSPREY keeps that
knowledge in three tiers, split by *when* the agent needs it and *what shape*
the knowledge has.

* **Always in context** -- the facility's core operating context, carried as
  Markdown rules under the project's ``.claude/rules/`` directory. These load
  into the main agent's context at the start of every session, the same way
  ``CLAUDE.md`` does.
* **Fetched on demand** -- the **Open Knowledge Format (OKF)** bundle, a
  directory of Markdown concept documents served by the
  ``osprey_facility_knowledge`` MCP server. Subsystem descriptions, device
  specs, procedures and physics notes stay out of context until a task calls
  for them.
* **Queried** -- the **facility graph**, which holds the machine's *structure*
  rather than prose: one node per device, the sections devices sit in, the
  classes they belong to, and the control-system addresses bound to each one.
  A specialist agent (``facility-knowledge-graph``) writes the Cypher, so you
  ask in plain language.

Each tier is documented separately:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Facility Rules
      :link: facility-rules
      :link-type: doc

      The always-in-context tier -- what the ``control-assistant`` preset ships
      in ``.claude/rules/`` and how to change it.

   .. grid-item-card:: The OKF Bundle
      :link: okf-bundle
      :link-type: doc

      The on-demand tier -- authoring, configuring, and serving the Open
      Knowledge Format bundle of concept documents.

   .. grid-item-card:: Search the Facility Graph
      :link: use-facility-graph
      :link-type: doc

      The queryable tier -- the questions the graph answers, how the specialist
      agent reaches it, and how its answers differ from the channel finder's.

.. toctree::
   :hidden:

   facility-rules
   okf-bundle
   use-facility-graph
