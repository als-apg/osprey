.. _how-to-channel-finder:

==============================
How to Use the Channel Finder
==============================

The Channel Finder translates natural language queries, such as "beam current"
or "RF cavity voltage", into control system addresses, such as
``SR:DIAG:DCCT:01:CURRENT:RB`` or ``SR:RF:CAVITY:01:VOLTAGE:RB`` in the
demo preset. It uses LLM-based pipelines, so a query can use
everyday terms rather than exact channel names.

Four pipelines do this, each by a different paradigm. They differ in the size
of the channel catalog they suit and in how far the search moves from the
channels as named to the channels as meant:

.. raw:: html
   :file: ../_diagrams/channel-finder-paradigms.html

The pipelines search the *curated* channels: the ones the facility description
declares and describes. A channel name met in an alarm, a script or a logbook
entry is often not one of them. For those names the Channel Finder also reads
the control system's own directory, which lists every channel the control
system serves: see :ref:`channel-finder-directory`.

.. seealso::

   T. Hellert et al., *Mach. Learn.: Sci. Technol.* **7**, 045010 (2026) ·
   `doi:10.1088/2632-2153/ae8219 <https://doi.org/10.1088/2632-2153/ae8219>`_.

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

That call is OSPREY's own request to the provider, not the agent's, so it needs
the provider's API key: a personal Claude login does not cover it
(:ref:`personal-login`). ``osprey build``, ``osprey validate`` and ``osprey web``
warn when the key is missing. The hierarchical, middle layer and graph
pipelines make no model request of their own. The agent drives their tools, so
they need no key beyond what the agent uses.

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
tree the build generates has the place levels, then ``class``, ``device``, and
a leaf per channel keyed by its signal; the leaf holds the address. The place
levels are the level words of the places that hold a channel or are an
ancestor of one, by the depth at which each first appears; a place tree with no
channel, such as a rack tree beside the ring, adds no level. The build stops
with ``view-unsupported`` when two of those level words first appear at one
depth, or one branch orders two against their depths; the stop names the trees
and the words. Use one level word per depth across them, keep channels out of
one tree, or select another ``channel_finder_mode``. Two channels of one device
that the tree would key alike, because one channel's address is another's
signal, also stop the build with ``view-unsupported``.

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
- A Family is a group of kind ``family``, the kind of a group that states
  none; a ``section`` is no Family. The devices of a
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


.. _channel-finder-directory:

The Channel Directory
=====================

Whichever pipeline runs, it answers only for the curated channels. An operator
who meets an unfamiliar name in an alarm or a script still wants to know what
it is and which server serves it. The channel directory answers that question.
It is the control system's own record of every channel it serves and of the
server process behind each one. OSPREY shows the names, descriptions and
properties exactly as the control system records them: nothing in the
directory is curated, and nothing that reads it can write.

At an EPICS facility, for example, the channels are PVs, the servers are IOCs,
and the directory is usually a ChannelFinder service. The ``demo`` preset
needs no such service, because its directory lists what the virtual
accelerator serves.

The directory is on when ``config.yml`` has a ``channel_directory`` section, and
the ``demo`` preset ships one. The same section can name an alarm server, an
alarm log and a put log, so that the alarms and the recorded writes of a
channel appear beside its record. :ref:`config-channel-directory` lists the keys
and shows how to switch the section off.

The section serves the agent and the operator from the same sources:

- The agent gains the ``channel-directory`` MCP server. With it the agent
  searches the directory, reads the record of a channel or of a server and,
  where their source is configured, reads alarms and recorded writes. Every
  tool is a read and runs without an approval prompt. The agent's prompt tells
  it to check the curated channels first and to say when a name is the control
  system's own. :doc:`/architecture/mcp-servers` lists the tools.
- The operator gains the directory view of the Channel Finder panel and, where
  their service is configured, the servers, Alarms and Archive views. See
  `The Directory, Servers, Alarms and Archive Views`_.

Every answer comes from the machine the control-target chip shows. On
*Simulator* the answers are the simulator's own channels and its replayed
alarm feed. On *Live* and *Rehearsal* they come from the facility's services.
When the current target has no source for an answer, the answer reads
``Not available for this target``. :doc:`/reference/connectors-api/services`
has the table of sources per target.

.. _channel-directory-words:

The control system's words
--------------------------

Every control system has its own names for a channel and for the server that
serves it, and the people who run the machine use those names. The panel and
the agent's answers use them too. OSPREY takes the words from the connector of
the control target an answer comes from:

==========================  =========  =============  ==============  ==============
Control system              A channel  A server       Directory view  Servers view
==========================  =========  =============  ==============  ==============
EPICS, simulator, stand-in  PV         IOC            All PVs         IOCs
TANGO                       attribute  device server  Attributes      Device servers
DOOCS                       property   server         Properties      Servers
Any other                   channel    server         Channels        Servers
==========================  =========  =============  ==============  ==============

The simulator and the stand-in use the EPICS words because both serve their
channels the way EPICS soft IOCs do. The words follow the target, so a TANGO
facility reads about PVs and IOCs while a session is on the simulator, and
about attributes and device servers on the live machine. Config keys, tool
names and commands are the same on every control system, and this page calls
the two views the directory view and the servers view.

A facility that writes its own connector states its words on the connector
class, in a ``channel_vocabulary`` attribute. The attribute is a mapping with
the keys ``channel``, ``channels``, ``server`` and ``servers``, and optionally
``directory`` and ``servers_view`` for the titles of the two views. A title
that is left out is the plural with a capital first letter. A connector
without the attribute gets the words of the last row.

.. _channel-directory-backends:

Directory backends
------------------

A directory backend reads one kind of directory service. Two ship with OSPREY:

- ``channelfinder`` reads a ChannelFinder service, the directory that EPICS
  facilities commonly run. Name it in ``channel_directory.type`` and give the
  address of the service in ``channel_directory.settings.url``.
- The simulator's backend lists every channel the virtual accelerator serves.
  It needs no configuration: OSPREY uses it on the simulator target whenever
  the section is present.

A facility whose directory is another service, such as a TANGO database,
writes a backend for it and names the class by its dotted path in
``channel_directory.type``. The backend implements the ``ChannelDirectory``
contract in :doc:`/reference/connectors-api/services`. The contract asks for a
search, the record of one channel and the values of a property. Listing the
servers is optional, and the server fleet depends on it. The alarm and
put-log sources of the section work the same way, and that page lists their
shipped backends too.


.. _channel-finder-server-fleet:

The Server Fleet
================

The directory says which servers exist. It does not say which of them are
running, and that is the first question when channels stop answering.
Answering it for a whole facility means reading a liveness channel on every
server, which is too many reads to repeat each time someone asks. The server
fleet service therefore reads them once per refresh period and keeps one
snapshot per control target. The servers view of the panel, the agent's
``server_fleet`` tool and the page of each server all answer from that
snapshot. The three always agree, and asking again costs the control system
nothing.

At an EPICS facility the servers are IOCs, the liveness channel is the
heartbeat PV of each IOC, and the panel titles the view **IOCs**
(:ref:`channel-directory-words`). The fleet reads through the connector of
each target and works with any directory backend that can list its servers,
which both shipped backends can (:ref:`channel-directory-backends`).

The service is off until ``config.yml`` has a ``channel_directory.fleet``
block; an empty one (``fleet: {}``) is enough. The ``demo`` preset runs the
service with a 10 s period, and :doc:`/getting-started/demo` stops a server
to show how the view responds. :ref:`config-channel-directory` lists the keys.

Reading the servers view
------------------------

The header shows when the snapshot was taken and how old it is, whether the
read path works, and the outcome of the directory listing and of each roster
source. Below it, one tile per state and per finding carries its count.
Selecting a tile keeps only its rows, and so do words typed into the search
box. The table has one row per server: its name, its host, its state, how
long it has been in that state, its findings, its status in the directory and
the roster sources that name it. The worst rows come first: stalled, silent,
unreached with a finding, alive with a finding, unknown, unreached, alive.
**Group by** gathers the rows under the field that
``channel_directory.fleet.group_by`` names.

Each server has one state:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - State
     - Meaning
   * - ``alive``
     - The liveness channel answered and, where it is a counter, moved since
       the previous reading.
   * - ``stalled``
     - It answered, and the counter has not moved.
   * - ``silent``
     - It gives no answer now and has answered before through this read path.
   * - ``unreached``
     - It has never answered through this read path.
   * - ``unknown``
     - No state could be established. The read failed, the read path is down,
       the server has no liveness channel, a counter has only one reading so
       far, or the read path is unverified and every server that answered in
       the last cycle went quiet at once.

Neither ``unreached`` nor ``unknown`` says that a server stopped: both mean
that the service could not tell. A finding may stand beside the state:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Finding
     - Meaning
   * - ``degraded``
     - The server is alive and one of its health readings is outside its good
       range. A health channel that could not be read does not count.
   * - ``expected_running_not_alive``
     - A roster source expects the server to run, and it is stalled, silent or
       unreached.
   * - ``expected_stopped_alive``
     - A roster source expects the server to be stopped, and it is alive.
   * - ``alive_unexpected``
     - The server is alive, and no roster source expects it to run or to stop.

The three expectation findings appear only when a roster source is
configured (`Roster sources`_), and ``unknown`` never carries a finding.

When the service has no state to show, the view says why instead of listing
servers, and adds that this says nothing about whether any server runs:

- *No snapshot yet*: the first refresh of the target has not completed.
- *Target not refreshed*: the target is not in
  ``channel_directory.fleet.targets``, it is a virtual accelerator served in
  process, or the service cannot connect to it.
- *Cannot list the servers*: no listing of the directory has succeeded so far,
  or the directory backend cannot list its servers.
- *Snapshot stale*: the service stopped refreshing, and the snapshot is older
  than three refresh periods (or three cycles, when a cycle outlasts the
  period).
- *server fleet unavailable*: the service gave no usable answer.

The ``server_fleet`` tool answers the agent from the same snapshot, with the
counts, the meaning of each state and the rows that pass its filters for
state, finding, group, host or text. It makes the same statements, except
that it still returns the rows of a stale snapshot, marked as the states of
that time.

Which servers are listed
------------------------

Each target lists the servers of its own directory, including a server whose
channels are all inactive, and reads their liveness from that target. The
stand-in uses the directory of the facility, so a server whose liveness
channel the stand-in does not serve shows ``unreached``. A server that has
left the directory and every roster source stays in the view, marked
``not listed``, for as long as it answers and for
``channel_directory.fleet.retention_days`` after it goes quiet.

Before it reads any server, each refresh reads the probe channel of the
target, one channel that the target always serves. When the probe does not
answer, a silent server proves nothing, so every server is ``unknown`` and
the header reports the read path as down. A target with no probe channel is
reported as *unverified*.

With the fleet on, the page of a server and the ``server_record`` tool show
the row of the snapshot: the state, the findings and the time the liveness
channel was read. Neither makes a read of its own, so the answer can be up to
one refresh period old. Without a fresh row they say why, and that no state
is known for the server.

Load per refresh
----------------

Only the service reads liveness channels, so the load on the control system
does not grow with the number of users. For each refreshed target, one
refresh costs:

- one read of the probe, then one concurrent read of every liveness and
  health channel, or none when the probe does not answer;
- one audit record on the ``server_fleet`` surface
  (:doc:`/reference/contracts/audit-trail`);
- one read of each roster source, on ``live`` only.

The directory is queried less often: on the first refresh, on every tenth
after it and after a failed attempt. Each query lists the servers and runs
one paged search per liveness and health suffix.

As an example, take a light source with about 1,600 liveness and health
channels behind one gateway. At the default period of 30 s the service makes
about 53 reads per second, writes 2,880 audit records per day and lists the
directory every five minutes.

Roster sources
--------------

On its own the fleet knows only which servers answer. A roster source tells
it what the facility expects: which servers should run, which should be
stopped, and which class and group each belongs to. That expectation is what
turns a ``silent`` server into the finding ``expected_running_not_alive``.
Roster sources and the named classes of ``channel_directory.fleet.classes``
apply on ``live`` only. On the virtual accelerator and the stand-in, every
server is judged by the heartbeat channel that the directory lists for it.

A roster source is a small piece of Python that the facility writes: any
object with a ``name`` and a ``read(deadline)`` method that returns a list of
``RosterRow(name, host, class_name, expected, group=None)`` from
``osprey.services.channel_directory.roster``. ``expected`` is ``"running"``,
``"stopped"`` or ``None``, and ``read`` raises to report a failure. The
contract is public and may still change between releases.

``channel_directory.fleet.roster_sources`` lists the sources, each as a
``module:attribute`` reference to a source object or to a function without
arguments that returns one. The rows of all sources merge into one row per
server. A source that fails is shown as unavailable with its reason, and the
other sources still count.

``StaticRoster`` is an example to start from. It reads a YAML list of
servers and sits in ``docs/source/how-to/control-systems/examples/roster/``
with ``roster_static.yml`` beside it. With that directory importable,
``roster_sources: ["roster_static:roster"]`` uses it. To check a source of
your own, call ``check_roster_source(source)`` from the roster module: it
returns one line per problem, and none when the source conforms.


Web Interface
=============

The browser-based channel explorer is the Channel Finder panel of the web
terminal (``osprey web``).

The explorer is read-only. To correct a channel, edit ``provenance/fixes.yaml`` in the
facility tree and run ``osprey build``.

The explorer browses a channel database. On the graph pipeline there is no such
file, so its Explore view is a finder over the store instead. Type words into
the search box in the panel toolbar (``/`` focuses it) and it keeps the
channels that match all of them. Each word is matched against the address, the
description, the device or signal name, and the name or synonyms of the
device's class or any class above it. The view switch is at the toolbar's
left. It offers Explore alone, or Curated and the directory view when
``config.yml`` has a ``channel_directory`` section, with the servers, Alarms
and Archive views when their service is configured. There is no Feedback
view: the graph keeps no saved hints, and its answers are not captured for
review. The result count and the search box are at its right.

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

The Directory, Servers, Alarms and Archive Views
------------------------------------------------

With a ``channel_directory`` section (:ref:`channel-finder-directory`) the view
switch gains the directory view, which searches the directory. Its tab
carries the control system's own title (:ref:`channel-directory-words`). The
``demo`` preset runs on the simulator, so there the tab reads **All PVs**:

1. Click **All PVs**, type ``cavity temperature`` into the search box and press
   Enter. The rows are the channels whose description holds both words, each
   with its description, current value, alarm, type and server.
2. Click ``SR:RF:CAVITY:01:TEMPERATURE:RB`` to open the page of that channel.
   A banner shows a latched alarm from the replayed alarm feed of the demo,
   with its guidance.

Words match descriptions first, and names only when no description matched;
every word must match. A word holding ``*`` or ``?`` matches names as a glob,
and a leading ``=`` asks for the exact name. The sidebar filters the result,
and **show request** reveals the request that ran.

Each row carries one of two marks:

- ◆ **Curated**: the facility description declares the channel.
- ◇ **Control system only**: the control system serves it and the facility
  description does not declare it.

The search words carry over from the Curated view. A ``Not here?`` line under
its results opens the directory view with the same words, and a curated
search with no hits offers to search the whole directory instead. The names
of ticked rows can be copied, sent to the assistant or, when CANOPY serves
Trace, plotted.

The page of a channel has one card per answer: Value, Alarm, Put history,
Directory, Archive and Logbook. A card whose service is not configured is one
dashed line. Clicking a server opens the page of that server, with its
channels, the alarms on it, its health channels and a chip that says whether
it is alive. The chip reads the heartbeat channel twice, 2 s apart, or shows
the state from the fleet snapshot when ``channel_directory.fleet`` is set
(:ref:`channel-finder-server-fleet`).

Three more views appear when their service is configured, and none of them
writes:

- The servers view lists every server of the control target with its liveness
  state and findings (:ref:`channel-finder-server-fleet`).
- **Alarms** lists what is in alarm now on the control target, worst first.
  To acknowledge an alarm, use the client of the alarm server.
- **Archive** says what the archive of the control target records: how many
  of the channels in the directory, and its newest sample.


The ``config.yml`` keys for every pipeline, and how the active one is served to
the agent, are in :doc:`/reference/contracts/channel-finder`.
