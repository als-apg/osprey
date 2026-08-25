Architecture
============

OSPREY deploys agentic AI in safety-critical control system environments, such as
particle accelerators. It uses **the Osprey agent** as the orchestrator, **MCP servers** as
the tool interface, and **pluggable connectors** for protocol-agnostic hardware access.

.. figure:: /_static/resources/architecture.png
   :alt: Osprey system architecture — from operator to facility, with the safety gate and approval workflow in-line.
   :align: center
   :width: 100%

   Osprey system architecture — from operator to facility, with the safety gate and approval workflow in-line.

Six pages break that picture down — how a tool call is guarded, how a write
travels, how reading differs from writing, and the three subsystems with
architecture of their own:

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Safety Chain
      :link: safety-chain
      :link-type: doc
      :shadow: md

      Every tool call passes a chain of PreToolUse hooks before it reaches an
      MCP server — the kill switch, the limits database, and the human
      approval gate.

   .. grid-item-card:: Data Flow
      :link: data-flow
      :link-type: doc
      :shadow: md

      A control-system write from prompt to hardware, step by step — and
      where the three safety hooks fire along the way.

   .. grid-item-card:: Retrieval Paths
      :link: retrieval-paths
      :link-type: doc
      :shadow: md

      The three independent read stacks — ARIEL's Postgres, the qmd sidecar,
      and the embedding-free channel finder — and which of them needs an
      embedding provider on the host.

   .. grid-item-card:: MCP Servers
      :link: mcp-servers
      :link-type: doc
      :shadow: md

      The ten core stdio servers the agent reaches, grouped by what they do,
      with the complete tool list for each.

   .. grid-item-card:: Python Executor
      :link: python-executor
      :link-type: doc
      :shadow: md

      How agent-written Python actually runs: nine safety layers, the
      readonly/readwrite execution modes, and the zones executed code may
      not touch.

   .. grid-item-card:: Virtual Accelerator
      :link: virtual-accelerator
      :link-type: doc
      :shadow: md

      A whole facility on real EPICS with LUME-backed physics — the layer
      map, the two transports, and the model seam that makes the physics
      replaceable.

.. seealso::

   :doc:`/how-to/control-systems/use-connectors`
      How to add a custom control system connector.

   :doc:`/how-to/deploy-project/index`
      How to create and deploy an OSPREY project.

.. toctree::
   :hidden:

   safety-chain
   data-flow
   retrieval-paths
   mcp-servers
   python-executor
   virtual-accelerator
