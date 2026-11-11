.. _retrieval-paths:

Retrieval Paths
===============

The :doc:`safety chain <safety-chain>` is how OSPREY *writes*. Reading is a
different shape: the agent answers questions from four retrieval stacks, each
with its own index and its own answer to whether embeddings are involved at all.

.. figure:: /_static/screenshots/retrieval_map_light.png
   :class: only-light
   :figclass: only-light
   :width: 100%
   :alt: The four retrieval stacks — the ARIEL Postgres stack, the qmd sidecar, the embedding-free channel finder and the facility knowledge graph — each feeding MCP tools the OSPREY agent calls.

   Every retrieval path, left to right: sources, ingest-time processing,
   indexes, query paths, and the agent.

.. figure:: /_static/screenshots/retrieval_map_dark.png
   :class: only-dark
   :figclass: only-dark
   :width: 100%
   :alt: The four retrieval stacks — the ARIEL Postgres stack, the qmd sidecar, the embedding-free channel finder and the facility knowledge graph — each feeding MCP tools the OSPREY agent calls.

   Every retrieval path, left to right: sources, ingest-time processing,
   indexes, query paths, and the agent.

Four things are worth reading off the map:

**Only one stack needs an embedding provider on the host.** :doc:`ARIEL's
</how-to/ariel/search-modes>` ``semantic`` mode embeds both the corpus at ingest
and the *query* at query time, so Ollama (or OpenAI) has to be reachable when an
operator searches, not just when entries are ingested. The qmd sidecar (a
search service that indexes a markdown corpus and answers combined keyword and
semantic queries) carries its own embedder inside the image. The channel finder
and the facility graph use no embeddings anywhere. The finder ranks with BM25
(a standard keyword-relevance score) or lets the agent walk a hierarchy. The
graph answers Cypher (Neo4j's query language), with a substring index for
phrases.

**The qmd sidecar is the one cross-stack component at query time.** It answers
ARIEL's ``hybrid`` mode and backs :doc:`facility-knowledge </how-to/facility-knowledge/index>`
search. A deployment runs one sidecar per corpus — one for the markdown mirror
ARIEL exports, one for the
:doc:`Open Knowledge Format (OKF) </how-to/facility-knowledge/okf-bundle>`
bundle, and one for each other corpus it declares. Its internals, and the
``rerank`` tradeoff, are covered under :ref:`qmd-search-sidecar`.

**The facility graph answers questions about relations.** The other stacks find
a text or an address. The :doc:`graph </how-to/facility-knowledge/use-facility-graph>`
relates devices, classes, sections and addresses to each other, so it can count
every magnet or list a section in beam order. Two subagents read it: the
facility-knowledge-graph agent for structure, and the channel finder in graph
mode to turn a phrase into an address. The channel database and the graph both
derive from the facility file's records, so they describe the same machine.

**Stacks degrade differently.** The dashed edges are fallbacks: OKF search drops
to a substring scan when the sidecar is absent, and ARIEL's ``semantic`` mode
hard-depends on a reachable provider. The graph has no fallback for Cypher: with
the store down, a query returns an error naming the remedy, while graph mode's
``search_channels`` still answers from its index.
