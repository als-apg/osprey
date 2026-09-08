================
Agent Interfaces
================

The web terminal is not the only way to reach the OSPREY agent. The same agent,
working on the same project, can be driven from a shell, started by an external
event, or answered from a chat room your team already sits in -- what differs
between those is who or what starts a run, and how much of it a human sees
while it happens. A fourth page is not an interface at all: it covers handing
the agent new tools, which changes what any of the others can do.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Run the Agent from the Command Line
      :link: cli-agent
      :link-type: doc
      :shadow: md

      ``osprey chat`` for an interactive session in your native terminal, and
      ``osprey query`` for a single headless prompt that exits with a
      meaningful code.

   .. grid-item-card:: Event Dispatch
      :link: event-dispatch
      :link-type: doc
      :shadow: md

      Turn webhooks and cron ticks into headless agent runs: the dispatcher,
      the worker, the triggers you author, and the two bearer tokens that
      guard them.

   .. grid-item-card:: Add an MCP Server
      :link: add-mcp-server
      :link-type: doc
      :shadow: md

      Give the agent new tools: declare an external server under
      ``mcp_servers:`` in your build profile, or -- as a contribution to
      Osprey itself -- write a framework server in Python.

   .. grid-item-card:: Chat Bridges
      :link: chat-bridges/index
      :link-type: doc
      :shadow: md

      Let a team ask questions from Nextcloud Talk or Google Chat and get the
      answer -- plots and files included -- back in the same conversation.

.. toctree::
   :hidden:

   cli-agent
   event-dispatch
   add-mcp-server
   chat-bridges/index
