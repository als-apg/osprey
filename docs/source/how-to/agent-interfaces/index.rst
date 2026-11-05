================
Agent Interfaces
================

Choose how you reach the OSPREY agent besides the web terminal: from the
command line, from an external event, or from a team chat room. It is the same
agent on the same project each way. What differs is who or what starts a run,
and how much of it a human sees while it happens. To give the agent new tools,
see :doc:`add-mcp-server`.

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

      Turn webhooks, cron ticks and EPICS channel crossings into headless
      agent runs: the dispatcher, the worker, the triggers you author, and the
      two bearer tokens that guard them.

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

      Let a team ask questions from Nextcloud Talk, Google Chat or Microsoft
      Teams and get the answer -- plots and files included -- back in the same
      conversation.

.. toctree::
   :hidden:

   cli-agent
   event-dispatch
   add-mcp-server
   chat-bridges/index
