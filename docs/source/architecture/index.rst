Architecture Overview
=====================

OSPREY deploys agentic AI in safety-critical control system environments — particle accelerators,
fusion experiments, and beamlines. It uses **the Osprey agent** as the orchestrator, **MCP servers** as
the tool interface, and **pluggable connectors** for protocol-agnostic hardware access.

.. figure:: /_static/resources/architecture.png
   :alt: Osprey system architecture — from operator to facility, with the safety gate and approval workflow in-line.
   :align: center
   :width: 100%

   Osprey system architecture — from operator to facility, with the safety gate and approval workflow in-line.


Safety Chain
------------

Every tool call the agent makes through the harness passes through a configurable chain of
**PreToolUse hooks** before reaching the MCP server. OSPREY's MCP servers are stdio child
processes the harness starts --- they expose no network port of their own --- so every tool call
that reaches one has already passed the hook chain. That guarantee belongs to the harness: a
process on the host that runs the server command itself is outside it, which is why host access
is part of the trust boundary. The chain for ``channel_write`` — the most safety-critical tool — has three stages:

.. raw:: html
   :file: ../_diagrams/safety-chain.html

1. **osprey_writes_check** — Kill switch. Blocks all writes when ``control_system.writes_enabled``
   is ``false`` in ``config.yml``. Applies to both ``channel_write`` and ``execute``.

2. **osprey_limits** — Validates the setpoint against the channel limits database
   (min, max, step size, writable flag). Only applies to ``channel_write``.

3. **osprey_approval** — Human approval gate. Per-tool policy dispatch: ``always`` (require
   approval every time), ``selective`` (ask the Osprey agent to decide), or ``skip``.


Data Flow
---------

A typical control system write follows this path. The three safety hooks fire between the Osprey agent
and the MCP server:

.. raw:: html
   :file: ../_diagrams/write-sequence.html


.. _retrieval-paths:

Retrieval Paths
---------------

The path above is how OSPREY *writes*. Reading is a different shape: the agent
answers questions from three independent retrieval stacks, each with its own
index and its own answer to whether embeddings are involved at all.

.. figure:: /_static/screenshots/retrieval_map_light.png
   :class: only-light
   :figclass: only-light
   :width: 100%
   :alt: The three retrieval stacks — the ARIEL Postgres stack, the qmd sidecar, and the embedding-free channel finder — each feeding MCP tools the Osprey agent calls.

   Every retrieval path, left to right: sources, ingest-time processing,
   indexes, query paths, and the agent.

.. figure:: /_static/screenshots/retrieval_map_dark.png
   :class: only-dark
   :figclass: only-dark
   :width: 100%
   :alt: The three retrieval stacks — the ARIEL Postgres stack, the qmd sidecar, and the embedding-free channel finder — each feeding MCP tools the Osprey agent calls.

   Every retrieval path, left to right: sources, ingest-time processing,
   indexes, query paths, and the agent.

Three things are worth reading off the map:

**Only one stack needs an embedding provider on the host.** :doc:`ARIEL's
</how-to/ariel/search-modes>` ``semantic`` mode embeds both the corpus at ingest
and the *query* at query time, so Ollama (or OpenAI) has to be reachable when an
operator searches, not just when entries are ingested. The qmd sidecar carries
its own embedder inside the image, and the channel finder uses no embeddings
anywhere — it ranks with BM25 and lets the agent walk a hierarchy.

**The qmd sidecar is the one cross-stack component.** It answers ARIEL's
``hybrid`` mode and backs :doc:`facility-knowledge </how-to/facility-knowledge/index>`
search, indexing both corpora — the markdown mirror ARIEL exports and the OKF
bundle — in one process. Its internals, and the ``rerank`` tradeoff, are covered
under :ref:`qmd-search-sidecar`.

**Every stack degrades rather than fails.** The dashed edges are fallbacks: OKF
search drops to a substring scan when the sidecar is absent, and ARIEL's
``semantic`` mode is the one path that hard-depends on a reachable provider.

.. seealso::

   :doc:`mcp-servers`
      Complete list of MCP servers and their tools.

   :doc:`virtual-accelerator`
      The Virtual Accelerator: its layers, its two transports, and the
      LUME model seam.

   :doc:`/how-to/control-systems/use-connectors`
      How to add a custom control system connector.

   :doc:`/how-to/deploy-project/index`
      How to create and deploy an OSPREY project.

.. toctree::
   :hidden:

   mcp-servers
   python-executor
   virtual-accelerator
