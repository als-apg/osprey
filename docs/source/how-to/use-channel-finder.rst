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
reads what the one before it wrote, so the order is the whole workflow. An
export that also carries the facility's lattice adds a fourth, ``verify``, and
a virtual accelerator to check --- see `From an Export to a Virtual
Accelerator`_ below.

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

An export written by ``mml_export.m`` 2.0 has three more files beside its
``ao.json``, and naming the ``ao.json`` brings them in too; they land as
``data/mml/lattice/<system>.mat``, ``data/mml/va.json`` and
``data/mml/response.json``. `From an Export to a Virtual Accelerator`_ is what
they are for.

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


From an Export to a Virtual Accelerator
---------------------------------------

Version 2.0 of the exporter saves more than the channel names. It also saves
the lattice deck the Middle Layer models the ring with, the conversion between
each family's hardware units and physics, the settings the machine sits at, and
an orbit response matrix for the ring, measured on the machine or computed
from a model, as the export records. With those four things the same workflow
also builds your deployment a **virtual accelerator**: a simulated ring that
answers your facility's own channel names, so the OSPREY agent can be exercised
against your addresses without beam.

None of that physics is written into OSPREY. Every family the model drives, it
drives because your export said so and you agreed. Two documents are what you
read to agree: the **VA MAP card**, which the OSPREY agent draws while you are
deciding, and ``VA-REPORT.md``, which ``verify`` writes once the model exists.

**Import brings the lattice in.** Run ``mml_export.m`` 2.0 once per sub-machine
as before. It now writes five files rather than two, and naming the ``ao.json``
imports all five:

.. code-block:: bash

   osprey mml import mymachine.storagering.ao.json

Two rules keep the directory honest. A deck is filed only when it is the ring
the export states it was sampled over --- the export records its energy, its
element count, a digest of its element names and where its ring parameters sit,
and a deck that disagrees on any of those is refused, naming the fact that
disagreed, rather than being modelled quietly. And an import replaces the last
one whole: the decks and files left by the export you are replacing are
removed, and the report names each one. ``data/mml/`` is the last import and
nothing else.

**Map decides what the model drives.** ``map --init`` writes one more block,
``virtual_accelerator:``, with a verdict for every family: ``couple``, naming
what it drives and how its units convert, or ``latch``, naming why it drives
nothing. Most families decide themselves --- a quadrupole drives a gradient, a
corrector drives a kick, a beam monitor reads an orbit --- and a family that
drives nothing says why:

.. code-block:: yaml

   BEND:
     verdict: couple
     kind: energy
     calibration: table
     nominal_source: Setpoint
   DCCT:
     verdict: latch
     reason: no lattice element

Where no rule can decide, the family carries one slot: the question written
out, an empty answer, and the words that answer it. There are three kinds.

- **A lattice type the table does not know** (``attype``) --- the export names
  a type OSPREY has no branch for, such as a septum. Answer ``latch`` to leave
  the family out of the model, or say what it drives:
  ``strength:<PolynomB|PolynomA>[<i>]``, ``kick:<0|1>``, ``energy``, ``rf`` or
  ``monitor:<x|y>``.
- **A field two families both drive** (``shared_field``) --- both bind the same
  field of the same element, and only one of them can own it. Answer
  ``owner:<family>`` naming the one that does, or ``latch`` for neither.
- **A family the Middle Layer reaches through code of its own**
  (``escape_hatch``) --- the export says a special function or a parameter
  group stands between the family and the lattice, so what it really does is
  not readable from the export. Answer ``latch`` to leave it out, or
  ``ignore_hook`` to bind it the ordinary way anyway.

``map --check`` refuses while any slot is still empty, exactly as it does for
an unanswered judgment, so nothing is emitted from a question nobody answered.

You do not have to read the block to answer it. During a guided install the
OSPREY agent draws the whole thing as one **VA MAP card**: what the export
brought and which families MATLAB itself refused, then the open questions one
row each with the question and its answer words verbatim, then the families it
would couple, then the families it would latch grouped by the reason they
share. It asks you to confirm the map before anything is emitted, and redraws
it after every answer (:doc:`/getting-started/osprey-install`). Re-running
``map --init`` on a tree that already has the block stops rather than
overwriting your answers: it names the file and tells you to pass
``--force-va``, which replaces the block.

**Emit writes the model.** The same ``mml emit`` run that writes the channel
database also writes five files for the virtual accelerator: the deck it runs
(``data/simulation/lattice.json``), which channel drives which element and how
(``data/simulation/va_bindings.json``), the machine it stands for
(``data/simulation/machine.json``), the channels carrying its state
(``data/machine_state_channels.json``), and the write band of every coupled
setpoint, in the deployment's shared ``data/channel_limits.json``. Emit stamps
each band it writes there, leaves every other entry byte-for-byte, and refuses
an address that file already bands differently without that stamp. That
collision can only be known once all five documents have been prepared, so when
it happens the five virtual-accelerator files are withheld while the channel
database, ontology and knowledge pages of the same run are already on the tree
--- the corpus, which emit writes last, is not; fix or remove the entries it
named and run emit again.

For a beam position monitor the bindings also carry the calibration your
control system states for that reading --- its gain, offset, roll and crunch,
one set per device, and only the ones your facility states. Nothing in the
simulation applies them: they are carried, not modelled, and a reading served
today is the same number with them as without.

An export still at 1.0 is not an error. Emit says ``VA lane skipped:
data/mml/va.json is not in the tree; re-export with mml_export 2.0 to enable
it``, and says ``carries no virtual accelerator for an imported system`` in
place of that middle clause when the file is there but keys no imported system.
Either way it writes everything else as it always did --- and takes any ring
the tree was serving with it. A harvest onto a deployment that ships its own
demo model removes ``data/simulation/lattice.json`` and
``data/simulation/va_bindings.json``, and says so: the export describes no
machine, so there is none to serve, and the demo's ring left in place would be
served over your own channel names. ``osprey build`` then reports
``VA_LATTICE=none``. A demo deployment you never harvested onto keeps its ring
and goes on serving it.

Scenarios are held against the machine that resolves them. A scenario under
``data/simulation/scenarios/`` names channels in its overrides and its archiver
events, and the simulation will not boot on one naming a channel
``data/simulation/machine.json`` does not carry. Emit asks that question of the
machine your deployment will serve: the one this run writes, when the export
carries a virtual accelerator, and otherwise the one already on the tree. So a
2.0 harvest refuses the demo's scenarios --- your machine has replaced the
demo's and their channels are gone with it --- while a 1.0 harvest leaves them
alone, because the machine they were written for is still the one being served.
Refused scenarios are named with the channels they ask for, in one ``rm`` line,
and so is a bundle the simulation could not read at all. Your own scenarios are
left alone as long as the served machine can resolve them.

**Verify checks the model against the machine.**

.. code-block:: bash

   osprey mml verify

Verify steers the model's correctors the way the facility steered its own,
reads the orbit that comes back, and compares it with the matrix the export
carries. An entry agrees when ``|R_model - R_file| <= 0.05 * max(|R_file|, 0.1
* rms(column))`` --- within five per cent of the exported value, or of a tenth
of that corrector column's own scale, whichever is the larger, so an entry near
zero is judged on size alone. Above that floor the sign has to agree too: a
corrector that moves the beam the wrong way is wrong however small the number
is.

It writes ``data/mml/VA-REPORT.md``, and that report is the second thing to
read. It opens with the verdict --- how many entries are inside the band and
how many agree in sign --- then where and at what energy the matrix was
measured, the worst disagreements for each monitor-and-corrector pair, the rows
it could not compare, the write bands the model needed widened, and the
nominals the model does not hold. Rows are matched to the export's device list
by sector and device rather than by position, and a row the export marks down
is dropped and named, so a disagreement in the report is a disagreement about
physics and not about bookkeeping.

**Build serves it.** Nothing the four verbs wrote is running yet:

.. code-block:: bash

   osprey build

The build copies the emitted tree into the deployment and writes ``VA_LATTICE``
into the project ``.env`` for it: the name of the lattice file in the built
tree, or ``none`` for a deployment that serves no model. The value is a file
name or that one word, so the service either runs the deck your export brought
or runs none (:doc:`/how-to/deploy-project/env-chain`). See
:doc:`/how-to/control-systems/use-virtual-accelerator` for running it.


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
