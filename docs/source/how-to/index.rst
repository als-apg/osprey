How-To Guides
=============

Task-oriented guides that walk you through common OSPREY operations step by step.
Each guide focuses on a single goal and assumes you already have a working OSPREY installation.
The sections follow the natural journey: build and deploy a project, operate the
agent day to day, then work with the bundled facility services.

Build & deploy
--------------

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Build Profiles
      :link: build-profiles
      :link-type: doc

      Build a facility-specific assistant from a profile you own — convention
      directories, taking ownership of framework artifacts, personas, and secrets.

   .. grid-item-card:: Deploy a Facility
      :link: deploy-a-facility
      :link-type: doc

      The end-to-end walkthrough: one deployment repo from ``osprey init``
      through the CI scaffolding to a running three-service stack.

   .. grid-item-card:: Container Deployment
      :link: deploy-project/index
      :link-type: doc

      The service stack behind a running project — compose templates, networking,
      the environment chain, and the agent's own container image.

   .. grid-item-card:: LLM Providers
      :link: llm-providers/index
      :link-type: doc

      Pick the provider that drives the Osprey agent and map the model tiers
      each one serves, including open-weight and self-hosted models behind the
      translation proxy.

Operate
-------

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Web Terminal
      :link: web-terminal/index
      :link-type: doc

      The browser cockpit for the Osprey agent — launching and theming it,
      adding your own side panels, reporting problems from inside it, and
      serving a whole team from one host.

   .. grid-item-card:: Agent Interfaces
      :link: agent-interfaces/index
      :link-type: doc

      The CLI agent, event-driven dispatch, external MCP servers, and chat
      bridges — the four ways to reach the agent.

   .. grid-item-card:: Health and Monitoring
      :link: health-and-monitoring/index
      :link-type: doc

      Is it up, and what is it doing — facility health checks you can extend,
      plus the agent's own logs and metrics over OTLP.

   .. grid-item-card:: Control Systems
      :link: control-systems/index
      :link-type: doc

      Connectors for your control system, the virtual accelerator to rehearse
      on, switching to the live machine, and what the agent may not touch.

Facility services
-----------------

.. grid:: 1 1 2 3
   :gutter: 3

   .. grid-item-card:: Use the Channel Finder
      :link: use-channel-finder
      :link-type: doc

      Search, filter, and explore control system channels using the Channel Finder
      service and its web interface.

   .. grid-item-card:: Facility Knowledge
      :link: facility-knowledge/index
      :link-type: doc

      The Open Knowledge Format bundle, the facility graph, facility rules, and
      the search sidecar that serves all of it to the agent on demand.

   .. grid-item-card:: ARIEL Logbook Search
      :link: ariel/index
      :link-type: doc

      Search over facility electronic logbooks with keyword and
      semantic retrieval modes, plus multi-step reasoning delegated to the
      Osprey agent.

   .. grid-item-card:: Bluesky Plans
      :link: bluesky/index
      :link-type: doc

      Run measurement plans through a durable queue — compose with the Osprey
      agent, review, start and stop in the BLUESKY panel, and add plans
      of your own.

.. toctree::
   :hidden:

   build-profiles
   deploy-a-facility
   deploy-project/index
   llm-providers/index
   web-terminal/index
   agent-interfaces/index
   health-and-monitoring/index
   control-systems/index
   use-channel-finder
   facility-knowledge/index
   ariel/index
   bluesky/index
