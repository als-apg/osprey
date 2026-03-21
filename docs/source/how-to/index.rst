How-To Guides
=============

Task-oriented guides that walk you through common OSPREY operations step by step.
Each guide focuses on a single goal and assumes you already have a working OSPREY installation.

Framework & Infrastructure
---------------------------

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Configure LLM Providers
      :link: configure-providers
      :link-type: doc

      Set up and switch between supported LLM providers — Anthropic, OpenAI, Google,
      CBORG, AMSC, Ollama, and others — via ``config.yml``.

   .. grid-item-card:: Deploy a Project
      :link: deploy-project
      :link-type: doc

      Create, configure, and deploy an OSPREY project from ``osprey init`` through
      ``osprey deploy`` to a running instance.

   .. grid-item-card:: Build Profiles
      :link: build-profiles
      :link-type: doc

      Assemble facility-specific assistants from templates with config overrides,
      file overlays, and custom MCP servers.

   .. grid-item-card:: Add an MCP Server
      :link: add-mcp-server
      :link-type: doc

      Build and register a new FastMCP server to expose domain-specific tools that
      Claude Code can discover and call.

   .. grid-item-card:: Use the Web Terminal
      :link: use-web-terminal
      :link-type: doc

      Launch and operate the Web Terminal interface for interactive Claude Code
      sessions with your control system.

   .. grid-item-card:: Use the Python Executor
      :link: use-python-executor
      :link-type: doc

      Run Claude-generated Python scripts safely in a containerized environment with
      access to the OSPREY runtime API.

   .. grid-item-card:: CLI Reference
      :link: /cli-reference/index
      :link-type: doc

      Complete reference for all ``osprey`` commands — init, deploy, config,
      health, claude, web, and more.

Services & Connectors
---------------------

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Add a Control System Connector
      :link: add-connector
      :link-type: doc

      Create a custom connector to integrate a new control system protocol (beyond EPICS
      and Mock) with OSPREY's protocol-agnostic architecture.

   .. grid-item-card:: Use the Channel Finder
      :link: use-channel-finder
      :link-type: doc

      Search, filter, and explore control system channels using the Channel Finder
      service and its web interface.

   .. grid-item-card:: ARIEL Logbook Search
      :link: ariel/index
      :link-type: doc

      Intelligent search over facility electronic logbooks with keyword, semantic,
      RAG, and agentic retrieval modes.


.. toctree::
   :hidden:

   configure-providers
   deploy-project
   build-profiles
   add-mcp-server
   use-web-terminal
   use-python-executor
   add-connector
   use-channel-finder
   ariel/index
   /cli-reference/index
