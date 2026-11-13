.. _how-to-channel-finder:

==============================
How to Use the Channel Finder
==============================

The Channel Finder translates natural language queries, such as "beam current"
or "RF cavity voltage", into control system addresses, such as
``SR:DIAG:DCCT:01:CURRENT:RB`` or ``SR:RF:CAVITY:01:VOLTAGE:RB`` in the
demo preset. It uses LLM-based pipelines, so a query can use
everyday terms rather than exact channel names.

The pipelines search the *curated* channels: the ones the facility description
declares and describes. A PV name met in an alarm, a script or a logbook entry
is often not one of them. For those the Channel Finder also reads the control
system's own directory, which lists every PV the control system announces: see
:ref:`channel-finder-all-pvs`.

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

The four are not size tiers of one design. Each takes a differently shaped
database and resolves a query by a different mechanism:

.. raw:: html
   :file: ../_diagrams/channel-finder-pipelines.html


In-Context Pipeline
===================

Loads the entire channel database into the LLM context for direct semantic
matching.

**How it works:** a single inner-LLM call. The complete channel database is
embedded in the system prompt. The model returns the most relevant channels in
one shot, with no query-splitting or iterative-correction stage.

``osprey build`` writes the database from the facility description as
``data/channel_finder/in_context.json``. It holds one row per channel, or per
channel of the subset the facility tags ``in_context``, with its name, address
and description. See :doc:`/reference/contracts/channel-finder` for the schema.
A facility description with no channel stops the build with
``view-unsupported``.

Validate and preview:

.. code-block:: bash

   osprey channel-finder validate
   osprey channel-finder preview


Hierarchical Pipeline
=====================

Navigates a tree level by level, with LLM-guided selection at each level. The
tree the build generates has the facility's place levels, then ``class``,
``device``, and a leaf per channel keyed by its signal; the leaf holds the
address. Two channels of one device that the tree would key alike, because one
channel's address is another's signal, stop the build with
``view-unsupported``.

The database declares the levels the pipeline navigates --- see
:doc:`/reference/contracts/channel-finder` for the schema.

Validate and preview:

.. code-block:: bash

   osprey channel-finder validate
   osprey channel-finder preview --depth 4 --sections tree,stats


Middle Layer Pipeline
=====================

A ReAct (reason and act) agent explores the database using query tools
(``list_systems``, ``list_families``, ``inspect_fields``,
``list_channels``, ``get_common_names``, ``statistics``, ``validate``, and —
when DuckDB is installed — ``run_sql``). What ``run_sql`` hands back is bounded
by ``channel_finder.query_max_rows`` (default 500). A longer result is cut and
flagged, naming the key, so the agent narrows the query instead of presenting a
partial list. The cap is on the agent's context, not on the database, which is
why it is set per deployment.

The database follows MATLAB Middle Layer (MML) functional organization
(system, then family, then field, then the channel names). ``osprey build``
writes it from the facility description as
``data/channel_finder/middle_layer.json``. The DuckDB copy ``run_sql`` reads
sits beside it as ``data/channel_finder/middle_layer.duckdb``. In that
database:

- A System is a top place.
- A Family is a group: every group of the facility is one. The devices of a
  class that no single-class group holds form a Family named by the class
  under their top place.
- A Field is a channel's signal, one channel per member, described by the
  vocabulary's sentence for that signal.

A channel whose signal some member lacks or has twice, or that has no signal,
is keyed by its address. A channel belongs to every Family that holds its
device. A channel listed under several Fields of one Family is found under each
of them.

Two things stop the build with ``view-unsupported``:

- a facility description with no group and no classed device;
- a channel whose address is the signal another channel's Field in the same
  Family is keyed by.

A facility that runs a Middle Layer
already has that structure: ``osprey facility import mml``
(:doc:`/how-to/import-mml-export`) brings it into the facility description.


Graph Pipeline
==============

Searches the facility knowledge graph instead of a channel database. A
graph-mode project ships no channel database: ``osprey build`` writes the graph
view ``data/graph/facility.ttl`` from the facility file. The channel finder
subagent finds addresses in one of two ways:

- by keyword lookup in the search index the build writes from that view;
- by writing read-only Cypher against the ``graphdb`` store seeded from it.

**How it works:** five tools, served under the ``channel-finder`` name like
every other pipeline's:

``capabilities``
   Reports how addresses are spelled and what prose the corpus carries.

``search_channels``
   Looks a phrase up in the search index and returns matching addresses a page
   at a time, with facet counts for section, system, class, signal and
   direction.

``example_queries``
   Returns runnable Cypher for the common channel questions, each with
   framework-default parameter values. The seed-time snapshot may replace those
   values with values captured from this corpus.

``get_schema``
   Lists the labels, relationship types and property names *this* graph
   actually holds.

``read_cypher``
   Runs one query and returns rows.

There is no resolution API behind them — the subagent looks a
phrase up or adapts an example rather than calling a lookup.

Configuration is the mode plus the store, and nothing else — no ``tier``,
because the pipeline has no tiered artifacts:

.. code-block:: yaml

   channel_finder_mode: graph

The render that field produces has no graph entry under
``channel_finder.pipelines``: that section comes out empty in graph mode,
because graph names no database file.

The search index the finder reads needs no key. The build writes it to
``./data/channel_databases/graph.duckdb`` beside the render's ``config.yml``,
which is why the mode and the store are the whole of the configuration.

The store comes from the ``services.graphdb`` block, either one this deployment
runs or one the facility already hosts (an explicit ``services.graphdb.uri``
and ``username``, with ``GRAPHDB_PASSWORD`` in the project ``.env``). See
:ref:`profile-graph-mode` for both shapes, and :doc:`deploy-project/index` for the
block itself. A build that enables the channel finder but renders no
``services.graphdb`` block is refused, naming the missing block, rather than
shipping a pipeline with nothing to read. The channel-finder login runs on its
host deployment's services, so it reads this mode from the host's
``services.graphdb`` block.

``osprey up`` loads the build's graph view into the store when the deployment
runs the store; a store the facility hosts holds whatever the facility loaded
into it. See :doc:`facility-knowledge/use-facility-graph` for what the graph
holds.

**What the subagent can search** depends on the corpus. On the build's graph
view a phrase can be matched against:

- the description written for a single channel;
- the prose for a group of devices or a top place;
- the synonyms an operator would say out loud.

A corpus the profile names in place of the view (``services.graphdb.ttl_path``)
may carry less prose. There the way in is a name, an alternate name, a section
or a device class.

``validate`` and ``preview`` have no channel database to open on this pipeline,
so both report what the store is and which commands act on it.

Health reports the store and the search index instead of a database
(:doc:`health-and-monitoring/configure-health-checks`):

- whether the store is reachable and how many resources it holds;
- whether the index is there and was built from the corpus the store was
  seeded with.

The web explorer opens on this pipeline too: its Explore view searches the
index rather than browsing a channel tree, and the device card reads the store.
See `Web Interface`_ below.


.. _channel-finder-all-pvs:

All PVs: the Control System's Directory
=======================================

Whichever pipeline runs, it answers only for the curated channels. The
directory answers for the rest. It is the control system's own record of every
PV and IOC, such as a ChannelFinder service, or the virtual accelerator's PV
list in the ``demo`` preset. Its names, descriptions and properties are shown
as the control system records them; nothing in it is curated, and nothing that
reads it can write.

The directory is on when ``config.yml`` has a ``pv_directory`` section; the
``demo`` preset ships one. The same section can name an alarm server, an alarm
log and a put log. :ref:`config-pv-directory` has the keys and how to switch
the section off.

**For the agent** the section adds the ``pv-directory`` MCP server: search the
directory, read a PV's or an IOC's record and, where their source is
configured, the alarms and recorded writes. Every tool is a read and runs
without an approval prompt. The agent's prompt tells it to check the curated
channels first and to say when a name is the control system's own.
:doc:`/architecture/mcp-servers` lists the tools.

**For the operator** it adds the All PVs, Alarms and Archive views of the
panel: see `All PVs, Alarms and Archive`_.

Every answer comes from the machine the control-target chip shows: the
simulator's own PVs and replayed alarm feed on *Simulator*, the facility's
services on *Live* and *Rehearsal*. An answer with no source for the current
target reads ``Not available for this target``.
:doc:`/reference/connectors-api/services` has the table of sources per target.


Web Interface
=============

The browser-based channel explorer is the Channel Finder panel of the web
terminal (``osprey web``).

The explorer is read-only. To correct a channel, edit ``fixes.yaml`` in the
facility tree and run ``osprey build``.

The explorer browses a channel database. On the graph pipeline there is no such
file, so its Explore view is a finder over the store instead. Type words into
the search box in the panel toolbar (``/`` focuses it) and it keeps the
channels that match all of them. Each word is matched against the address, the
description, the device or signal name, and the name or synonyms of the
device's class or any class above it. The view switch is at the toolbar's
left: Explore and Feedback, or Curated, All PVs and Feedback when ``config.yml``
has a ``pv_directory`` section, with Alarms and Archive when their service is
configured. The result count and the search box are at its right.

The sidebar narrows further:

- Section, System and Signal count matching channels.
- Device class counts devices and rolls each class up over its subclasses.
- Direction splits them into read, write, read/write and undirected.

A class that groups devices without being a kind of device itself
is shown in muted italic. Every facet is counted with its own filter lifted, so
a number says what a second pick in that facet would add. Each facet folds from
its heading and clears with its ✕. Active filters show as chips you can click
to remove, and when a facet list is capped the panel says so. The sidebar
button at the left end of the toolbar shows and hides the sidebar. In a narrow
panel the sidebar opens over the results. The ⋯ button at the right end of the
toolbar switches the row density, chooses and resets the columns, and clears
the filters or the selection. Inside the terminal, these entries are in the ⋯
menu of the tile, under This panel.

Results come fifty to a page: device, section, address, direction, signal and
description. Click a header to sort on it, drag it to move the column, drag its
right edge to resize it, and double-click that edge to fit it to its content.
The header's right-click menu also hides a column, fits all of them, or resets
the layout. Order, widths and hidden columns are remembered per browser.

Clicking a device name opens its card beside the results and marks its rows.
The card shows where the device sits in the machine, and every channel bound to
it grouped by signal. Escape closes it.

Tick the rows you want, then:

- **Copy addresses** puts them on the clipboard one per line.
- **Send to assistant** puts them into the prompt on one line for you to send.
  It is offered only when the panel runs inside the terminal.

The panel never submits anything itself.

Right-click a row, a facet value, a chip or a channel in the card for the
actions that apply there:

- copy the address, device name or row;
- send to the assistant;
- open the device;
- narrow the search to that row's section, system or signal.

Right-click the empty results area to clear the filters or the selection, or to
choose columns. Shift+F10 or the menu key opens the same menu from the
keyboard. The
table is one Tab stop: arrows, Home and End move between rows, Space selects,
Enter opens the device and ⌘C/Ctrl+C copies the address. A focused header sorts
on Enter and moves with Alt+←/→.

The strip below the results names:

- the corpus file and the store it was loaded into;
- the tools the OSPREY agent queries that same store with;
- the counts of devices, channels, classes, signals and sections, read live
  from the store.

If the store is unreachable the view says so and offers a Retry. If it is
reachable but empty, it names the ``osprey build && osprey up`` command that
fills it. Channel validation is not offered on this pipeline. The
channel-suggestion typeahead in the web panels still works in graph mode:
``osprey build`` reads the channel names from the facility file and writes them
into the snapshot the panels use.

All PVs, Alarms and Archive
---------------------------

With a ``pv_directory`` section (:ref:`channel-finder-all-pvs`) the view switch
gains **All PVs**, which searches the directory. In the ``demo`` preset:

1. Click **All PVs**, type ``cavity temperature`` into the search box and press
   Enter. The rows are the cavities' temperature readbacks, each with its
   description, current value, alarm severity, record type and IOC.
2. Click ``SR:RF:CAVITY:01:TEMPERATURE:RB`` to open its PV page. A banner
   shows a latched alarm from the demo's replayed alarm feed, with its
   guidance.

Words match descriptions first, and names only when no description matched;
every word must match. A word holding ``*`` or ``?`` matches names as a glob,
and a leading ``=`` asks for the exact name. The sidebar filters the result,
and **show request** reveals the request that ran.

Each row carries one of two marks:

- ◆ **Curated**: the facility description declares the PV.
- ◇ **Control system only**: the control system announces it and the facility
  description does not.

The search words carry over from the Curated view: a ``Not here?`` line under
its results opens All PVs with the same words, and a curated search with no
hits offers to search the whole directory instead. The names of ticked rows
can be copied, sent to the assistant or, when CANOPY serves Trace, plotted.

A **PV page** has one card per answer: Value, Alarm, Put history, Directory,
Archive and Logbook. A card whose service is not configured is one dashed
line. Clicking an IOC opens its **IOC page**: its PVs, the alarms on it, its
health PVs, and an alive chip that reads the heartbeat PV twice, 2 s apart.

Two more views appear when their service is configured. Neither writes:

- **Alarms** lists what is in alarm now on the control target, worst first.
  Acknowledge in the alarm server's own client.
- **Archive** says what the control target's archive records: how many of the
  directory's PVs, and its newest sample.


The ``config.yml`` keys for every pipeline, and how the active one is served to
the agent, are in :doc:`/reference/contracts/channel-finder`.
