=========
Contracts
=========

A contract is the shape two sides agree on: the JSON a service returns, the
parameters a tool takes, the behaviour every implementation of an interface has
to honour. These pages are what you read when you consume one of those shapes
or write something new against it -- a CI job that parses a health report, a
connector for a control system OSPREY does not ship, a script that loads a
channel database. For the day-to-day work of configuring any of it, the how-to
guides are the better starting point.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: The ``osprey health --json`` Contract
      :link: health-json
      :link-type: doc
      :shadow: md

      The exact shape of the JSON document ``osprey health --json`` writes, the
      exit codes that go with it, and how to consume both from a CI job.

   .. grid-item-card:: Connector Contracts
      :link: connectors
      :link-type: doc
      :shadow: md

      What every connector must implement at its edges: how large array values
      come back, how a write reports whether it took effect and which safety
      gates it passes first, and the archiver's historical-data contract.

   .. grid-item-card:: Python Executor Contract
      :link: python-executor
      :link-type: doc
      :shadow: md

      The ``execute`` and ``execute_file`` MCP tool parameters, the JSON a
      successful run returns, and how a failing one reports itself.

   .. grid-item-card:: ARIEL Contracts
      :link: ariel
      :link-type: doc
      :shadow: md

      The MCP tools that reach the logbook search service, the structure of a
      search result, the capabilities endpoint the web interface reads, and the
      database schema every entry is stored in.

   .. grid-item-card:: Channel Finder Contracts
      :link: channel-finder
      :link-type: doc
      :shadow: md

      The JSON schema each channel database pipeline expects, the
      ``config.yml`` keys that select a pipeline and point it at its database,
      and how the active pipeline is served to the agent as one
      ``channel-finder`` MCP server.

   .. grid-item-card:: Facility Graph Contracts
      :link: facility-graph
      :link-type: doc
      :shadow: md

      What the facility graph stores, how its names are spelled, and what each
      of the four ``graph`` MCP tools returns.

   .. grid-item-card:: Audit Trail Contract
      :link: audit-trail
      :link-type: doc
      :shadow: md

      Which file under ``var/audit/`` each safety decision lands in, the
      fields of every record, who can read the files, and what the trail does
      not promise.

.. toctree::
   :hidden:

   health-json
   connectors
   python-executor
   ariel
   channel-finder
   facility-graph
   audit-trail
