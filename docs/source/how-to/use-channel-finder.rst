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

Set the active pipeline with the top-level ``channel_finder_mode:`` field in
the deployment's ``profile.yml``:

.. code-block:: yaml

   channel_finder_mode: in_context  # or "hierarchical", "middle_layer", "graph"

or from the command line, which writes the same line:

.. code-block:: bash

   osprey set channel_finder_mode=in_context

The build renders that field into ``channel_finder.pipeline_mode`` and the
per-mode ``channel_finder.pipelines`` block in ``config.yml``. Both are the
build's to write: a ``config:`` line for either is refused, naming the field to
set instead (:ref:`profile-derived-keys`).

When the field is unset, OSPREY auto-detects: it uses the first
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
when DuckDB is installed — ``run_sql``). What ``run_sql`` hands back is bounded
by ``channel_finder.query_max_rows`` (default 500): a longer result is cut and
flagged, naming the key, so the agent narrows the query instead of presenting a
partial list. The cap is on the agent's context, not on the database, which is
why it is set per deployment.

The database follows MATLAB Middle Layer (MML) functional organization
(system, then family, then field, then the channel names). A facility that runs
a Middle Layer already has that structure, and ``osprey mml`` installs from it.


Installing from a Middle Layer
------------------------------

Three verbs, run in the deployment repository: ``import`` reads the export,
``map`` records what it means, ``emit`` writes the deployment's files. Each one
reads what the one before it wrote, so the order is the whole workflow.

**Import.** First export the Middle Layer as JSON on the MATLAB machine. The
exporter ships with OSPREY --- ``osprey scaffold pull
control-assistant:data/mml/mml_export.m`` puts it in ``data/mml/``, and the
README beside it covers the MATLAB half. Run it once per sub-machine, then read
the exports in:

.. code-block:: bash

   osprey mml import mymachine.storagering.ao.json mymachine.booster.ao.json

They become ``data/mml/ao.json``, ``data/mml/ad.json`` and ``PROFILE.md``, a
census of what arrived: systems, families, distinct addresses, and how many
signals it could not tell a readback from a setpoint by. The location is fixed
rather than a flag, because the later verbs read the same directory. An export
that does not name its own sub-machine is given a system name with ``--system
TOKEN``.

**Map.** Nothing about your facility is guessed at silently. ``mml map --init``
writes ``data/mml/mapping.yaml``, a skeleton with a slot for every decision:
what each raw family is called in the deployment, what kind of device it is,
which section of the machine it sits in, and whether each signal is read or
written.

.. code-block:: bash

   osprey mml map --init

Fill the empty slots and review the ones the skeleton guessed --- the OSPREY
agent can propose all of them, but the wording is yours to confirm, because
this file is what the agent will later read your machine through.

A few slots are questions rather than wording. Where the export leaves the
shape of a family genuinely ambiguous, the skeleton writes a ``judgments:``
block with one null slot per question, and ``PROFILE.md`` lists them family by
family, naming the devices and PVs each one concerns. There are three kinds:

- **Rows beyond the devices.** A field carries more channel rows than the
  family has devices --- a DCCT whose monitor lists an average current, a
  lifetime and a total. Each extra row is answered ``drop`` to leave it out,
  ``device`` to make it one more device of the family, or ``field: Lifetime``
  to give it a field of its own. ``device`` adds a device to the family, which
  every broadcast field also reaches.
- **A device bound by no channel.** The export lists the device, but no channel
  names it --- a third tune reading the machine does not measure. Answer
  ``drop`` to leave it out, or ``keep`` to keep it as a device with no channel.
- **A PV shared across devices.** One PV stands for a whole group, typically
  magnets fed from one supply. Answer ``keep_all`` to keep the PV on every
  member, or name the single device that owns it, as
  ``{<lowest ordinal>: <owning ordinal>}`` the way ``PROFILE.md`` spells the
  slot. Naming an owner can leave a member with no channel of its own; that
  member stays a device, like a kept unbound one, and ``PROFILE.md`` says how
  many members that is before you answer.

Then check the file back against the export:

.. code-block:: bash

   osprey mml map --check

The check names every problem it finds and exits non-zero while any remain --- a
judgment left null, or answered in a way the export cannot carry, is one of
them. It also says how many guessed slots are still unreviewed. Adding
``--no-derived`` turns each of those into a problem of its own --- the stricter
run to pass before you go live.

**Emit.** The last verb writes the deployment's files from the export and the
checked mapping:

.. code-block:: bash

   osprey mml emit --duckdb

That is the middle-layer database at
``data/channel_databases/middle_layer.json``, the facility ontology, the
knowledge pages under ``data/facility_knowledge/``, and a Turtle corpus named
for your facility. ``--duckdb`` also writes the DuckDB copy that the
``run_sql`` tool reads. That copy holds one row per process variable, so two
slots naming the same PV --- a shared setpoint, or one row broadcast to every
device in a family --- become a single row; emit names each of them as it
writes, and the middle-layer database and the corpus keep every binding. Emit
refuses before writing anything while a judgment is unanswered or impossible,
naming each one and sending you back to ``map --check``. A project still
carrying the demo facility's databases
or knowledge pages is refused before anything is written, with one ``rm`` line
naming exactly what to remove; run it and emit again. Finally ``osprey build``
copies the emitted files into the deployment --- the running stack keeps its
old copy until then.

**One export, either paradigm.** Emit writes the middle-layer database *and*
the corpus every time, because both describe the same machine. Which one the
channel finder uses is the ``channel_finder_mode`` field: leave it at
``middle_layer`` to query the database, or set it to ``graph`` and point
``services.graphdb.ttl_path`` at the corpus
(:doc:`/how-to/facility-knowledge/use-facility-graph`). The unchosen file stays
in the repository, so trying the other paradigm later is a configuration
change, not a second install.


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

Configuration is the mode plus the store, and nothing else — no ``tier``,
because the pipeline has no tiered artifacts:

.. code-block:: yaml

   channel_finder_mode: graph

The render that field produces has no graph entry under
``channel_finder.pipelines``: that section comes out empty in graph mode,
because graph names no database file.

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
``channel-finder-standalone`` preset, which ships no such block, cannot run
this mode.

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
