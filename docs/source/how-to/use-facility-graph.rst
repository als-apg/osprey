.. _how-to-facility-graph:

=========================
Search the Facility Graph
=========================

The facility graph holds the *structure* of the machine: the devices, the
sections they sit in, the classes they belong to, and the control system
addresses bound to each one. It is served by the ``graphdb`` store
(:doc:`deploy-project`), and the OSPREY agent searches it by writing Cypher —
Neo4j's query language, the graph counterpart of SQL — through the ``graph``
MCP server. You never write it yourself.

It answers a different kind of question than the :doc:`channel finder
<use-channel-finder>`. The channel finder turns a phrase into an address —
"beam current" into ``SR:DIAG:DCCT:01:CURRENT:RB``. The graph relates addresses
to each other: how many correctors the storage ring has, what sits in a section
in beam order, which PVs one device exposes and which of them are written, or
which devices share a PV.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - What the graph holds and how its names are spelled
   - The four tools and the order to use them in
   - Why the server can only read, and what it refuses before dialing
   - What each degraded state means and the remedy it names
   - How to generate a graph corpus from your own channel database

   **Prerequisites:** a project with a ``services.graphdb`` block — the
   ``control-assistant`` preset ships one, already seeded.


What the Graph Holds
====================

The corpus is a Turtle (``.ttl``) file in the NARAD convention — facts written
as subject-predicate-object triples — which the neosemantics plugin turns into
the nodes and relationships a query walks. Four kinds of thing matter to a
query:

* **Devices** — one node per physical device, carrying ``sourceName``,
  ``sectionCode``, ``sPositionM`` (position along the beamline) and
  ``ordinalInSection``.
* **Channel bindings** — one node per control system address, carrying
  ``fullPv``, ``protocol`` and ``confidence``. A device reaches its bindings
  over ``HASBINDING``.
* **Signals** — what a binding reads or writes, reached over ``READSSIGNAL`` or
  ``WRITESSIGNAL``. Exactly one of the two sits on every binding, so the
  direction of an address is a property of the graph rather than a guess from
  its name.
* **Classes** — the device ontology, linked by ``SUBCLASSOF``. A device is
  typed by ``TYPE``. This is what lets "every magnet" find an ``HCorrector``
  without the query naming ``HCorrector``.

Two spelling rules come from neosemantics and catch out anyone writing Cypher
from memory:

* Every node carries the label ``Resource`` plus the local name of its RDF
  class — ``(b:ChannelBinding)`` narrows to bindings, ``(c:Class)`` to ontology
  classes.
* Relationship types are **uppercased**: ``HASBINDING``, not ``hasBinding``.
  Property names keep their original spelling. A query naming something that
  does not exist returns zero rows rather than an error, so a guess produces a
  confident wrong answer — which is what ``get_schema`` is for.


The Four Tools
==============

The ``graph`` server exposes four tools. Used in this order they cost the
fewest turns:

1. ``example_queries`` — a curated, runnable set covering the common question
   shapes: device counts by class (concrete and rolled up the ontology), every
   device of a class including its subclasses, a section walk in beam order,
   every PV of one device split by direction, the class hierarchy as a table
   and as inheritance chains, a read/write split across all bindings, and
   devices sharing one PV. Each carries its Cypher and a parameter set for each
   shipped corpus. Adapting one is reliable where inventing a query is
   guesswork.
2. ``get_schema`` — the labels, relationship types and per-label property names
   actually present in *this* graph, plus the NARAD namespace prefix map for
   reading and building ``uri`` values. Call it when you need a name the
   examples do not use. Property lists are sampled rather than exhaustive;
   labels and relationship types are complete.
3. ``read_cypher`` — runs one read-only query and returns
   ``columns``/``rows``/``row_count``/``truncated``. Pass values through
   ``params`` as ``$name`` placeholders rather than pasting them into the query
   text; the curated examples are written that way, so an example plus its
   parameter set runs unedited.
4. ``capabilities`` — the static manifest (description, tool list, operating
   notes). It does not dial the store, so a successful reply says nothing about
   whether the graph is up.

Results are bounded in two directions. At most
``services.graphdb.query_max_rows`` rows come back — the reply says
``truncated: true`` when more matched — and the store cancels a query that
outlives ``services.graphdb.query_timeout_s``. Both bound one *question*, not
the store. Raising them spends the agent's context window rather than the
store's memory: a few thousand rows crowd out the conversation long before they
trouble Neo4j. The better answer to a truncated result is usually a narrower
query — add a filter, bound the traversal, or aggregate.


Read-Only by Construction
=========================

Nothing the agent passes can change the graph. Two independent layers say so:

* Every query runs inside a **read transaction**. A query that tries to write
  is refused by the store itself, and comes back as a validation error naming
  the read-only posture.
* A **query gate** runs before the store is dialed at all. It refuses extension
  procedures and functions — the ``apoc``, ``n10s``, ``db``, ``dbms`` and
  ``gds`` namespaces, whether called with ``CALL`` or used as a function in a
  ``RETURN`` — and refuses ``LOAD CSV``. Only two read procedures are
  allowlisted (``db.labels`` and ``db.relationshipTypes``, the ones
  ``get_schema`` itself uses). Ordinary ``CALL { … }`` subqueries pass, and the
  gate keeps scanning inside them.

The gate exists because the store's single account is write-capable and can
reach the network: an extension procedure or a CSV import would be a way around
both facts. The graph is rebuilt from its Turtle corpus, so a change to the
graph belongs in that file followed by ``osprey knowledge seed-graph``.


When the Graph Cannot Answer
============================

Each degraded state comes back distinctly, naming its own remedy, so the agent
can tell "no data" from "no store" and report the difference rather than
retrying. The first five are error envelopes; a truncated result is a normal
answer that says it was cut short.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - State
     - What it means, and the remedy it names
   * - **Not configured**
     - No ``services.graphdb`` block. Normally you never see this: without the
       block the server is not rendered into the project at all. It appears
       only when the server was force-enabled anyway (see below). Add the
       block, and list ``graphdb`` in ``deployed_services`` if this deployment
       should run the store.
   * - **Unreachable**
     - The store did not answer. Check that the ``graphdb`` service is up and
       start it with ``osprey up``; if it is running, check that
       ``services.graphdb.port_host`` — or ``services.graphdb.uri`` for an
       external store — names the address it is actually published on.
   * - **Authentication failed**
     - Both ends read ``GRAPHDB_PASSWORD``, and the store keeps the value the
       project ``.env`` carried when its data volume was first created. Restore
       that value in the project ``.env``, or reset the store's data volume so
       it is created again with the current one. The value is never echoed
       anywhere.
   * - **Empty**
     - The store is running but holds no corpus, so no query can match. Load it
       with ``osprey knowledge seed-graph``.
   * - **Timed out**
     - The query outlived ``services.graphdb.query_timeout_s`` and the store
       cancelled it. Narrow the query first; raise the key only for a query
       that is genuinely long-running.
   * - **Truncated**
     - More rows matched than ``services.graphdb.query_max_rows``. The reply
       says so and suggests a ``LIMIT``, an aggregate, or a narrower ``MATCH``.

Both bound keys live in the ``services.graphdb`` block, alongside the image and
port settings documented in :doc:`deploy-project`.

.. note::

   ``osprey health`` reports the same store from the outside, under its
   ``graphdb`` category — bolt reachability and how many resources the graph
   holds. See :doc:`configure-health-checks`.


Leaving the Graph Tools Out
===========================

The server renders wherever ``services.graphdb`` is configured. To keep the
store but take the tools away from the agent — to shrink the tool surface, or
while benchmarking something else — turn the server off explicitly:

.. code-block:: yaml

   claude_code:
     servers:
       graph:
         enabled: false

The override is read after the store check, so the opposite spelling
(``enabled: true``) forces the server on even with no ``services.graphdb``
block. In that build ``read_cypher`` and ``get_schema`` can only report *not
configured*, while ``example_queries`` and ``capabilities`` still answer from
their static content — which is why a reply from either says nothing about
whether a store exists. Useful for testing the wiring, not for anything else.


Generating a Corpus
===================

``osprey knowledge build-ttl`` derives a NARAD-convention Turtle corpus from a
project's own channel database, so the graph describes the same machine the
channel finder does:

.. code-block:: console

   $ osprey knowledge build-ttl data/demo_machine.ttl
   Wrote data/demo_machine.ttl.
     512 devices, 2908 channel bindings, 113 signals.
     direction from channel limits: data/channel_limits.json
     Load it with: osprey knowledge seed-graph data/demo_machine.ttl

One device node per device, one binding per address, one class per device
family, and a read or write direction on every signal. Inputs you do not name
come from the project's config:

``--channel-db``
   ``channel_finder.pipelines.hierarchical.database.path``, resolved against
   the ``config.yml`` directory — ``data/channel_databases/hierarchical.json``
   in a rendered project. The generator reads the **hierarchical**-paradigm
   database, the one whose addresses are
   ``RING:SYSTEM:FAMILY:DEVICE:FIELD:SUBFIELD``; a project built for another
   paradigm does not ship it, and the command says which database files it did
   find and points at this option.

``--limits``
   ``control_system.limits_checking.database_path``. This file is what tells a
   readback from a setpoint, so the corpus and the write-safety layer agree on
   which channels are written. When no limits file is configured, the address
   grammar decides instead — a ``:SP`` subfield writes, everything else reads.
   **Every run reports which of the two it used**, in the line shown above.

``--ontology``
   The FAMILY-to-class table. Defaults to the demo-machine table shipped with
   OSPREY; give your own when your device families are not the demo machine's.

To have every deployment seed the file you wrote, point
``services.graphdb.ttl_path`` at it. That key is deliberately *not* where the
output argument defaults to: it usually names a hand-curated corpus, and a bare
run of the verb would overwrite it.

Then load it (see :doc:`okf-bundle` for the seeding verb in full):

.. code-block:: console

   $ osprey knowledge seed-graph data/demo_machine.ttl


What the Presets Seed
=====================

* **control-assistant** seeds ``./data/demo_machine.ttl`` — the demo machine,
  generated from the preset's own channel database with the command above and
  regenerable the same way. The ALS corpus still ships alongside it at
  ``./services/graphdb/als_gtb.ttl``; point ``ttl_path`` there, or at a corpus
  of your own, to seed that instead.
* **ariel-standalone** seeds ``./services/graphdb/als_gtb.ttl``, a real
  facility's corpus, as the worked non-demo example.

.. note::

   A build profile that ships its own ``data:`` tree replaces the bundle's
   ``data/`` directory wholesale, which takes ``demo_machine.ttl`` with it.
   Such a profile has to set ``services.graphdb.ttl_path`` in its ``config:``
   overlay — at its own corpus, or at ``./services/graphdb/als_gtb.ttl``.
   Otherwise the deploy warns, the store comes up empty, and every query
   reports the empty state above.


One Machine, Several Views
==========================

On the ``control-assistant`` preset the graph and the channel finder are
generated from the same source, so their address spaces line up exactly — and
where a view is smaller, it is a strict subset rather than a different machine:

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - View
     - Addresses
     - Relationship to the graph
   * - Graph corpus (``demo_machine.ttl``)
     - 2,908
     - 512 devices; 396 of the bindings are write-direction, and they are
       exactly the ``:SP`` addresses.
   * - Channel finder, hierarchical and middle-layer builds
     - 2,908
     - The same set, address for address.
   * - Channel finder, in-context (tier-1) builds
     - 569
     - A strict subset — the tier-1 database is a smaller selection of the same
       machine, every address of which is in the graph.
   * - Virtual accelerator simulation
     - 1,036
     - A strict subset — the channels the simulated machine actually serves.

So on a tier-1 build the agent can find a device in the graph that the
in-context channel finder does not list, and on any build it can find an
address the virtual accelerator does not serve. Both are the documented shapes
of those smaller views, not a mismatch. A build with a channel database of your
own restores the correspondence by regenerating the corpus from it.


Multi-User Operator Terminals
=============================

The ``control-assistant-readonly`` and ``control-assistant-readwrite`` personas
(:doc:`multi-user`) get the same graph tools, reading the hosting deployment's
store. They are attached renders — they deploy no services of their own — so
they have to be told which port that store is published on. Per-user web
terminal containers run with ``network_mode: host``, so a container's
``localhost`` *is* the deployment host, and both presets pin the bolt port
directly:

.. code-block:: yaml

   config:
     services.graphdb.port_host: 7687

Two consequences worth knowing before you move anything:

* **Move the port on the hosting deployment and you must move the same number
  in both persona presets.** They carry their own copy; nothing derives it for
  them.
* A ``deployment.bind_address`` pinned to a specific non-loopback interface
  publishes the store there and not on ``localhost``, so the personas would no
  longer reach it. URL-backed panels have the same shape — they name
  ``http://127.0.0.1:<port>`` outright (:doc:`web-terminal/panels`) — so a
  deployment that moves off loopback has to revisit both.

The read-only persona receives ``GRAPHDB_PASSWORD`` as well. The store has a
single write-capable account, so read-only-ness here is enforced by the graph
server's read transaction rather than by the credential — the same posture the
ARIEL database credential already takes. ``control-assistant-ariel`` has no
control surface and no graph tools by design.


.. seealso::

   :doc:`deploy-project`
      The ``services.graphdb`` block: image, ports, corpus, memory, and the
      query bounds.

   :doc:`okf-bundle`
      The ``osprey knowledge`` CLI, including ``seed-graph`` and ``build-ttl``.

   :doc:`use-channel-finder`
      The other way into the same machine — phrases to addresses.

   :doc:`/architecture/mcp-servers`
      MCP servers provided by the framework, including ``graph``.
