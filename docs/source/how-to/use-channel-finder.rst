.. _how-to-channel-finder:

==============================
How to Use the Channel Finder
==============================

The Channel Finder translates natural language queries (e.g., "beam current,"
"terminal voltage") into control system addresses (e.g., ``SR:DCCT:Current``,
``TMVST``). It uses LLM-based pipelines, so a query can use everyday terms
rather than exact channel names.

.. seealso::

   Hellert et al. (2025), *From Natural Language to Control Signals*,
   `arXiv:2512.18779 <https://arxiv.org/abs/2512.18779>`_.

   :ref:`retrieval-paths`
      Where the Channel Finder sits among OSPREY's retrieval stacks --- it is
      the one that uses no embeddings at all.

   :doc:`/reference/contracts/channel-finder`
      The database JSON schemas, the ``config.yml`` keys, and how the active
      pipeline is served to the agent.


Choosing a Pipeline
===================

Set the active pipeline in ``config.yml``:

.. code-block:: yaml

   channel_finder:
     pipeline_mode: in_context  # or "hierarchical", "middle_layer" or "graph"

When ``pipeline_mode`` is unset, OSPREY auto-detects: it uses the first
pipeline that has a database configured, preferring middle layer, then
hierarchical, then in-context. Auto-detection never lands on ``graph`` — that
pipeline reads no database file, so there is nothing to detect and you name it
explicitly.

+---------------------------+----------------------------------------------+
| Pipeline                  | Best for                                     |
+===========================+==============================================+
| **In-Context**            | Small/medium systems (< few hundred channels)|
+---------------------------+----------------------------------------------+
| **Hierarchical**          | Large systems with strict naming patterns    |
+---------------------------+----------------------------------------------+
| **Middle Layer**          | Large systems organized by function (MML)    |
+---------------------------+----------------------------------------------+
| **Graph**                 | Machines described in a knowledge graph      |
+---------------------------+----------------------------------------------+

The three are not size tiers of one design. Each takes a differently shaped
database and resolves a query by a different mechanism:

.. raw:: html
   :file: ../_diagrams/channel-finder-pipelines.html


In-Context Pipeline
===================

Loads the entire channel database into the LLM context for direct semantic
matching.

**How it works:** a single inner-LLM call — the complete channel database is
embedded in the system prompt and the model returns the most relevant channels
in one shot (no query-splitting or iterative-correction stage).

The database is a flat JSON structure loaded by ``TemplateChannelDatabase``,
with standalone entries and template entries for device families --- see
:doc:`/reference/contracts/channel-finder` for the schema.

Build a database from CSV, then validate and preview:

.. code-block:: bash

   osprey channel-finder build-database --use-llm
   osprey channel-finder validate
   osprey channel-finder preview

.. note::

   ``build-database`` writes into the **profile** the project was built from
   (``processed/channel_database.json`` inside its ``data/`` tree), not into the
   project — a generated database belongs beside the inputs it came from, and
   survives a rebuild there. That deliberately marks the project stale; clear
   the advisory by rebuilding:

   .. code-block:: bash

      osprey channel-finder build-database
      osprey build

   The pipelines — and a bare ``validate`` / ``preview`` — read the database
   referenced in ``config.yml`` (under ``data/channel_databases/``). If you
   built to a different name, either point the commands at it with
   ``--database`` or update the config path; otherwise you are silently
   validating the old database.


Hierarchical Pipeline
=====================

Navigates a nested hierarchy (system, family, device, field, subfield) using
recursive LLM-guided selection at each level.

The database declares the levels the pipeline navigates and the naming pattern
the addresses follow --- see :doc:`/reference/contracts/channel-finder` for the
schema.

Advanced features: navigation-only levels, friendly names via
``_channel_part``, optional levels with ``_is_leaf``, and custom separators
via ``_separator``.

Validate and preview:

.. code-block:: bash

   osprey channel-finder validate
   osprey channel-finder preview --depth 4 --sections tree,stats


Middle Layer Pipeline
=====================

A React agent explores the database using query tools
(``list_systems``, ``list_families``, ``inspect_fields``,
``list_channels``, ``get_common_names``, ``statistics``, ``validate``, and —
when DuckDB is installed — ``run_sql``).

The database follows MATLAB Middle Layer (MML) functional organization
(System -> Family -> Field -> ChannelNames). Convert from MML exports:

.. code-block:: bash

   python -m osprey.services.channel_finder.utils.mml_converter \
      --input path/to/mml_exports.py:MML_ao_SR \
      --output data/channel_databases/middle_layer.json


Graph Pipeline
==============

Searches the facility knowledge graph instead of a channel database. A
graph-mode project ships no channel database: the corpus describes the machine,
and the channel finder subagent finds addresses either by keyword lookup in the
search index the build derives from that corpus, or by writing read-only Cypher
against the seeded ``graphdb`` store.

**How it works:** five tools, served under the ``channel-finder`` name like
every other pipeline's. ``capabilities`` reports how addresses are spelled and
what prose the corpus carries; ``search_channels`` looks a phrase up in the
search index and returns matching addresses a page at a time, with facet counts
for section, system, class, signal and direction; ``example_queries`` returns
runnable Cypher for the common channel questions, each with framework-default
parameter values that the seed-time snapshot may replace with values captured
from this corpus; ``get_schema`` lists the labels, relationship types and
property names *this* graph actually holds; ``read_cypher`` runs one query and
returns rows. There is no resolution API behind them — the subagent looks a
phrase up or adapts an example rather than calling a lookup.

Configuration is the mode plus the store, and nothing else. There is no graph
entry under ``pipelines:`` — that section renders empty in graph mode — and no
``tier``, because the pipeline has no tiered artifacts:

.. code-block:: yaml

   channel_finder:
     pipeline_mode: graph
     pipelines:        # nothing here — graph names no database file

The store block names one more thing: the search index the finder reads, under
``services.graphdb.index_path`` — ``./data/channel_databases/graph.duckdb``
unless the project says otherwise, which is why the mode and the store are the
whole of the configuration.

The store comes from the ``services.graphdb`` block, either one this deployment
runs or one the facility already hosts (an explicit ``services.graphdb.uri``
and ``username``, with ``GRAPHDB_PASSWORD`` in the project ``.env``). See
:ref:`profile-graph-mode` for both shapes, and :doc:`deploy-project/index` for the
block itself. A build that enables the channel finder but renders no
``services.graphdb`` block is refused, naming the missing block, rather than
shipping a pipeline with nothing to read — which is why the
``channel_finder_standalone`` app template, which carries no such block, cannot
run this mode.

Load the corpus into the store:

.. code-block:: bash

   osprey knowledge seed-graph data/demo_machine.ttl

``osprey up`` does that for you when the deployment runs the store; a store the
facility hosts holds whatever was loaded into it, so seed it deliberately. See
:doc:`facility-knowledge/use-facility-graph` for what the graph holds and how a corpus is
generated.

**What the subagent can search** depends on the corpus. On a corpus
``osprey knowledge build-ttl`` generated — the demo machine — a phrase can be
matched against the description written for a single channel, against what the
last two tokens of an address mean, against the prose for a device family, a
system or a ring, and against the synonyms an operator would say out loud. A
corpus imported from a facility export may carry less prose: there the way in
is a name, an alternate name, a section or a device class.

``validate`` and ``preview`` have no channel database to open on this pipeline,
so both report what the store is and which commands act on it. Health reports
the store and the search index instead of a database: whether the store is
reachable and how many resources it holds, and whether the index is there and
was built from the corpus the store was seeded with
(:doc:`health-and-monitoring/configure-health-checks`).

The web explorer opens on this pipeline too: its Explore view searches the
index rather than browsing a channel tree, and the device card reads the store.
See `Web Interface`_ below.


Web Interface
=============

Launch the browser-based channel explorer:

.. code-block:: bash

   osprey channel-finder web
   osprey channel-finder web --port 9000

The explorer browses a channel database. On the graph pipeline there is no such
file, so its Explore view is a finder over the store instead. Type words into
the search box and it keeps the channels that match all of them — against the
address, the description, the device or signal name, and the name or synonyms
of the device's class or any class above it. A facet rail narrows further:
Section, System and Signal count matching channels, Device class counts devices
and rolls each class up over its subclasses, and Direction splits them into
read, write, read/write and undirected. A class that groups devices without
being a kind of device itself is shown in muted italic. Every facet is counted with its own
filter lifted, so a number says what a second pick in that facet would add.
Active filters show as chips you can click to remove, and when a facet list is
capped the panel says so.

Results come fifty to a page: device, section, address, direction, signal and
description. Clicking a device name opens a card for it — where it sits in the
machine, and every channel bound to it grouped by signal. Tick the rows you
want, and **Copy addresses** puts them on the clipboard one per line, while
**Send to assistant**, offered only when the panel runs inside the terminal,
puts them into the prompt on one line for you to send. The panel never submits
anything itself.

The badge naming the corpus file and the store it was loaded into, the chips
naming the tools the OSPREY agent queries that same store with, and the header
counts of devices, channels, classes, signals and sections read live from the
store are all unchanged. If the store is unreachable the view says so and
offers a Retry, and if it is reachable but empty it names the ``osprey
knowledge seed-graph`` command that fills it. Channel validation is not offered
on this pipeline. The channel-suggestion typeahead in the web panels still
works in graph mode: ``osprey build`` reads the channel names out of the Turtle
corpus named by ``services.graphdb.ttl_path`` and writes them into the snapshot
the panels use.


The ``config.yml`` keys for every pipeline, and how the active one is served to
the agent, are in :doc:`/reference/contracts/channel-finder`.
