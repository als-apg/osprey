How-To Guides
=============

Task-oriented guides that walk you through common OSPREY operations step by step.
Each guide focuses on a single goal and assumes you already have a working OSPREY installation.
The sections follow the natural journey: build and deploy a project, run and
operate the agent, extend it for your facility, then explore the bundled
services and tutorials.

Build & Deploy a Project
------------------------

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Build Profiles
      :link: build-profiles
      :link-type: doc

      Assemble facility-specific assistants from templates with config overrides,
      file overlays, and custom MCP servers.

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

Run & Operate the Agent
-----------------------

.. grid:: 1 1 2 3
   :gutter: 3

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

   .. grid-item-card:: Configure Health Checks
      :link: configure-health-checks
      :link-type: doc

      Extend ``osprey health`` with facility probe checks and plugins, and tune
      the suite's cost classes and timeouts via the ``health:`` config block.

Extend & Integrate
------------------

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Add a Control System Connector
      :link: add-connector
      :link-type: doc

      Create a custom connector to integrate a new control system protocol (beyond EPICS
      and Mock) with OSPREY's protocol-agnostic architecture.

   .. grid-item-card:: Add an MCP Server
      :link: add-mcp-server
      :link-type: doc

      Build and register a new FastMCP server to expose domain-specific tools that
      the Osprey agent can discover and call.

   .. grid-item-card:: Use the Python Executor
      :link: use-python-executor
      :link-type: doc

      Run agent-generated Python scripts safely in a containerized environment with
      access to the OSPREY runtime API.

   .. grid-item-card:: Facility Knowledge
      :link: use-facility-knowledge
      :link-type: doc

      What the Open Knowledge Format is and why OSPREY stores facility knowledge
      as cross-linked markdown, plus how to structure, author, and serve a
      bundle to the agent on demand.

Bundled Services & Tutorials
----------------------------

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Use the Channel Finder
      :link: use-channel-finder
      :link-type: doc

      Search, filter, and explore control system channels using the Channel Finder
      service and its web interface.

   .. grid-item-card:: ARIEL Logbook Search
      :link: ariel/index
      :link-type: doc

      Search over facility electronic logbooks with keyword and
      semantic retrieval modes, plus multi-step reasoning delegated to the
      Osprey agent.

   .. grid-item-card:: Use the Virtual Accelerator
      :link: use-virtual-accelerator
      :link-type: doc

      Run the Control Assistant tutorial against a containerized PyAT soft-IOC that
      serves real EPICS Channel Access with live storage-ring physics.

   .. grid-item-card:: Agent-Assisted Grid Scan Tutorial
      :link: agent-assisted-grid-scan
      :link-type: doc

      Ask the Osprey agent to compose an n-dimensional grid scan, watch it fill
      the PLAN panel live, then Execute and watch points land in SCAN RESULTS.

.. seealso::

   :doc:`CLI Reference </cli-reference/index>` — complete reference for all
   ``osprey`` commands: build, deploy, config, health, claude, web, and more.

.. toctree::
   :hidden:

   build-profiles
   configure-providers
   run-open-models
   deploy-project
   containerize-project
   web-terminal/index
   use-cli-chat
   non_interactive_query
   event-dispatch
   monitor-agent
   configure-health-checks
   add-connector
   add-mcp-server
   use-python-executor
   use-facility-knowledge
   use-channel-finder
   ariel/index
   use-virtual-accelerator
   agent-assisted-grid-scan
   /cli-reference/index
