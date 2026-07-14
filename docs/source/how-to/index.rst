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
      CBORG, AMSC i2, Ollama, and others — via ``config.yml``.

   .. grid-item-card:: Run Open & Local Models
      :link: run-open-models
      :link-type: doc

      Drive the Osprey agent with open-weight or self-hosted models via the
      translation proxy, and benchmark their capability with ``scripts/benchmark/``.

   .. grid-item-card:: Deploy a Project
      :link: deploy-project
      :link-type: doc

      Create, configure, and deploy an OSPREY project from ``osprey build`` through
      ``osprey deploy`` to a running instance.

   .. grid-item-card:: Containerize a Project
      :link: containerize-project
      :link-type: doc

      Build and run the container image generated for every project — build args,
      path relocation, air-gapped mode, and Kubernetes notes.

   .. grid-item-card:: Build Profiles
      :link: build-profiles
      :link-type: doc

      Assemble facility-specific assistants from templates with config overrides,
      file overlays, and custom MCP servers.

   .. grid-item-card:: Add an MCP Server
      :link: add-mcp-server
      :link-type: doc

      Build and register a new FastMCP server to expose domain-specific tools that
      the Osprey agent can discover and call.

   .. grid-item-card:: Web Terminal
      :link: web-terminal/index
      :link-type: doc

      The browser cockpit for the Osprey agent — launching it, theming every
      OSPREY interface at once, and adding your own themed side panels.

   .. grid-item-card:: Use the CLI Chat Interface
      :link: use-cli-chat
      :link-type: doc

      Run the Osprey agent in your native terminal with companion services accessible
      in a browser.

   .. grid-item-card:: Non-Interactive Agent Queries
      :link: non_interactive_query
      :link-type: doc

      Run the OSPREY agent headlessly from CI pipelines and automated workflows
      with ``osprey query`` — read-only, structured JSON output, and clear exit codes.

   .. grid-item-card:: Use the Python Executor
      :link: use-python-executor
      :link-type: doc

      Run agent-generated Python scripts safely in a containerized environment with
      access to the OSPREY runtime API.

   .. grid-item-card:: Event Dispatch
      :link: event-dispatch
      :link-type: doc

      Turn external events — webhooks and cron ticks — into headless Osprey agent
      runs, deployed as containers or run locally.

   .. grid-item-card:: Monitor the Agent
      :link: monitor-agent
      :link-type: doc

      Emit the agent's logs and metrics over OTLP to any backend, or deploy the
      opt-in local OpenObserve store alongside your project.

   .. grid-item-card:: CLI Reference
      :link: /cli-reference/index
      :link-type: doc

      Complete reference for all ``osprey`` commands — build, deploy, config,
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

   .. grid-item-card:: Use the Virtual Accelerator
      :link: use-virtual-accelerator
      :link-type: doc

      Run the Control Assistant tutorial against a containerized PyAT soft-IOC that
      serves real EPICS Channel Access with live storage-ring physics.

   .. grid-item-card:: ARIEL Logbook Search
      :link: ariel/index
      :link-type: doc

      Search over facility electronic logbooks with keyword and
      semantic retrieval modes, plus multi-step reasoning delegated to the
      Osprey agent.

   .. grid-item-card:: Facility Knowledge
      :link: use-facility-knowledge
      :link-type: doc

      What the Open Knowledge Format is and why OSPREY stores facility knowledge
      as cross-linked markdown, plus how to structure, author, and serve a
      bundle to the agent on demand.


.. toctree::
   :hidden:

   configure-providers
   run-open-models
   deploy-project
   containerize-project
   build-profiles
   add-mcp-server
   web-terminal/index
   use-cli-chat
   non_interactive_query
   use-python-executor
   event-dispatch
   monitor-agent
   add-connector
   use-channel-finder
   use-virtual-accelerator
   ariel/index
   use-facility-knowledge
   /cli-reference/index
