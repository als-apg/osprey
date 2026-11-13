.. _reference-facility-graph:

==========================
Facility Graph Contracts
==========================

What the facility graph stores, how its names are spelled, and what each of the
four ``graph`` MCP tools returns. For the task of searching it, see
:doc:`/how-to/facility-knowledge/use-facility-graph`.


What the Graph Holds
====================

The corpus is a Turtle (``.ttl``) file in the NARAD convention — facts written
as subject-predicate-object triples — which the neosemantics plugin turns into
the nodes and relationships a query walks. Seven kinds of thing matter to a
query:

* **Places** — one node per place, carrying ``placePath`` (its id) and
  ``sectionCode``, and, unless it is a top place, one ``PARTOF`` to its parent
  place. A place has no RDF class, so its only label is ``Resource``.
* **Devices** — one node per physical device, carrying ``deviceId``,
  ``sourceName`` (the device's id), ``system``, ``placePath`` and ``sectionCode``, plus
  ``sPositionM`` (position along the beamline, in metres), ``lengthM``,
  ``ordinalInPlace`` and ``ordinalInModel``, each only when the facility file
  carries it, and the prose for the levels the device sits under:
  ``familyDescription`` and ``systemDescription``. A device with a place
  reaches it over ``LOCATEDIN``. Relation claims add edges between devices,
  places and groups (`Relations`_).
* **Channel bindings** — one node per control system address, carrying
  ``bindingId``, ``fullPv`` and ``description``, the sentence written for that
  one channel. A device reaches its bindings over ``HASBINDING``.
* **Signals** — what a binding reads or writes, reached over ``READSSIGNAL`` or
  ``WRITESSIGNAL``: a setpoint channel writes its signal and a readback channel
  reads it, as the channel's ``role`` in the facility file says. The queue
  server's settables and readables come from those same roles
  (:doc:`/how-to/bluesky/write-plans`).
* **Classes** — the device ontology, linked by ``SUBCLASSOF``. A device is
  typed by ``TYPE``. This is what lets "every magnet" find an ``HCorrector``
  without the query naming ``HCorrector``. Classes also carry the synonyms an
  operator would say out loud, as ``skos:altLabel``.
* **Claims** — one ``:Claim`` node per relation claim, holding the evidence
  behind its edge (`Relations`_).
* **Groups** — one ``:Group`` node per group that is an end of a written
  claim, carrying ``groupId`` and its names, and one ``HASMEMBER`` to each
  member device. A group no claim names is no node.

Places, devices, channel bindings and groups carry their names as
``altLabel``, a list; a place's or device's alias names (``scheme_names``) are
entries too. A name is an identifier, so match a whole entry, ignoring case — ``CONTAINS``
would let ``U1`` match ``U10``:

.. code-block:: cypher

   MATCH (n) WHERE NOT n:Class AND ANY(l IN n.altLabel WHERE toLower(l) = toLower($name))

The descriptions are what makes the graph reachable from a phrase rather than
only from a name: "the magnets that bend the beam" matches text no address
spells. They sit on bindings rather than on signals because a description is
written for one address, while a signal node is shared by every address that
reads or writes it.

The descriptions and ``system`` come from the facility file, so they are there
in the build's graph view and not necessarily in a corpus imported straight
from a facility's own export. On such a corpus the way in is
a name, an ``altLabel``, a section or a class.

Three spelling rules come from neosemantics and catch out anyone writing Cypher
from memory:

* Every node carries the label ``Resource`` plus the local name of its RDF
  class — ``(b:ChannelBinding)`` narrows to bindings, ``(c:Class)`` to ontology
  classes.
* Relationship types are **uppercased**: ``HASBINDING``, not ``hasBinding``.
  Property names keep their original spelling. A query naming something that
  does not exist returns zero rows rather than an error, so a guess produces a
  confident wrong answer — which is what ``get_schema`` is for.
* ``altLabel`` is a **list**, not a string: the store keeps every synonym
  instead of letting the last one win. Match a class synonym over
  ``(c:Class)`` with ``ANY(l IN c.altLabel WHERE toLower(l) CONTAINS $term)``.
  Never compare the list itself with ``=``: ``c.altLabel = $term`` is a list
  against a text and never matches, and the string operators do not apply to
  the list at all.


Relations
=========

Each relation claim in the facility file
(:doc:`/how-to/describe-your-facility`) is an edge from its subject end to its
object end, each end the device, place or group it names. The edge is one
relationship type per relation type:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Relation type
     - Relationship
   * - ``part_of``
     - ``PARTOF``
   * - ``located_in``
     - ``LOCATEDIN``
   * - ``powered_by``
     - ``POWEREDBY``
   * - ``cabled_to``
     - ``CABLEDTO``
   * - ``read_out_by``
     - ``READOUTBY``
   * - ``controlled_by``
     - ``CONTROLLEDBY``
   * - ``triggered_by``
     - ``TRIGGEREDBY``
   * - ``interlocked_by``
     - ``INTERLOCKEDBY``
   * - ``bounded_by``
     - ``BOUNDEDBY``
   * - ``networked_to``
     - ``NETWORKEDTO``
   * - ``supplied_by``
     - ``SUPPLIEDBY``
   * - ``same_as``
     - ``SAMEAS``
   * - ``runs_on``
     - ``RUNSON``

The claim's ``standing`` picks the edge. An ``accepted`` claim is the plain
relationship. A ``contested`` claim is its contested twin, the type with
``CONTESTED`` appended (``POWEREDBYCONTESTED``), so a walk over plain
relationships never crosses a disputed fact. A ``superseded`` claim is no edge.
Ports and units do not change the edge, and equal claims from several sources
are one edge. ``CABLEDTO``, ``NETWORKEDTO`` and ``SAMEAS`` are symmetric, one
edge per pair of ends: match them undirected, ``(a)-[:CABLEDTO]-(b)``.
``SAMEAS`` is a claim like any other; nothing merges two records on it.

``PARTOF`` and ``LOCATEDIN`` each gain a meaning. ``PARTOF`` also joins a
device to a device. A ``located_in`` claim naming a device's own place adds
nothing, but one naming another place is a second ``LOCATEDIN``, so a query
can get two rows for one device; the resolved place is the one its
``placePath`` names.

Each claim is also a ``:Claim`` node, the evidence behind its edge. It reaches
its two ends over ``CLAIMSUBJECT`` and ``CLAIMOBJECT``, and each rival that has
a claim node too over ``CONTESTS``. It carries these properties, each only when
the claim has it: ``claimId``, ``relationType``, ``basis`` (``stated`` when the
claim states none), ``rule``, ``standing``, ``document``, ``documentVersion``,
``documentState``, ``locator``, ``subjectPort``, ``objectPort``,
``subjectUnit`` and ``objectUnit``. Each scalar attribute is one more property
(a bool as ``"true"`` or ``"false"``), under its key when the key is an
identifier the graph does not already use, else under ``attribute_`` and the
key with each byte outside ``[A-Za-z0-9_]`` written ``_xHH_``: ``cable-id`` is
``attribute_cable_x2D_id``. A claim whose end is a group reaches the
``:Group`` node, which carries ``groupId`` and reaches each member device over
``HASMEMBER``.

Two settings of the deployment's config choose what is written:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Setting
     - Values
     - What it writes
   * - ``facility_graph.claims``
     - ``all`` (default), ``contested``, ``none``
     - Which claims get a ``:Claim`` node: every claim, the claims in a
       dispute (a non-empty ``contests``, so under a policy the winner and the
       superseded loser too), or none. The edges stay the same.
   * - ``facility_graph.basis``
     - ``all`` (default), ``stated``
     - Whether ``inferred`` claims and inferred alias names appear at all.

Any other value stops the render with a message naming the setting and its
values. Under ``basis: stated`` each kept claim shows the standing the build
stamped, so a stated claim an inferred rival contests stays contested, with
that rival out of the graph.


The Four Tools
==============

The ``graph`` server exposes four tools. Used in this order they cost the
fewest turns:

1. ``example_queries`` — a curated, runnable set covering the common question
   shapes: device counts by class (concrete and rolled up the ontology), every
   device of a class including its subclasses, a section walk in beam order,
   every PV of one device split by direction, the class hierarchy as a table
   and as inheritance chains, a read/write split across all bindings, and
   devices sharing one PV. Each carries its Cypher and a parameter set whose
   values are framework defaults taken from the shipped demo machine. Adapting
   one is reliable where inventing a query is guesswork.
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

In a deployed project the agent rarely needs the first two: the staging step
inside every ``osprey up``, which seeds or re-verifies the store, captures the
live store's schema (property
lists complete, not sampled) together with the curated examples, and bakes them
into the rendered ``facility-knowledge-graph`` agent prompt, stamped with the
seed marker's checksum. Each example's parameters are marked in the block as
captured from this corpus or as framework defaults, so the agent knows which
values it can trust as real addresses. The block also carries the class
vocabulary — every class's ``altLabel`` synonyms — read from the same store, so
the agent maps an operator's word to a class without a hard-coded list, and
goes straight to ``read_cypher``. Because the writer of the snapshot is the
writer of the store, prompt and graph cannot drift apart silently; a rebuilt
render resets the prompt to a placeholder, and the next ``up`` fills it back
in. The tools stay registered as the recovery path — a query that returns zero
rows for a name the snapshot lists means the store changed out from under it,
and ``get_schema`` is the arbiter.

Results are bounded in two directions. At most
``services.graphdb.query_max_rows`` rows come back — the reply says
``truncated: true`` when more matched — and the store cancels a query that
outlives ``services.graphdb.query_timeout_s``. Both bound one *question*, not
the store. Raising them spends the agent's context window rather than the
store's memory: a few thousand rows crowd out the conversation long before they
trouble Neo4j. The better answer to a truncated result is usually a narrower
query — add a filter, bound the traversal, or aggregate.
