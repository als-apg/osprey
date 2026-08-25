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
the nodes and relationships a query walks. Four kinds of thing matter to a
query:

* **Devices** — one node per physical device, carrying ``sourceName``,
  ``sectionCode``, ``system``, ``sPositionM`` (position along the beamline) and
  ``ordinalInSection``, plus the prose for the three levels the device sits
  under: ``ringDescription``, ``systemDescription`` and ``familyDescription``.
* **Channel bindings** — one node per control system address, carrying
  ``fullPv``, ``protocol``, ``confidence`` and three texts: ``description`` is
  the sentence written for that one channel, while ``fieldDescription`` and
  ``subfieldDescription`` say what the last two tokens of the address mean. A
  device reaches its bindings over ``HASBINDING``.
* **Signals** — what a binding reads or writes, reached over ``READSSIGNAL`` or
  ``WRITESSIGNAL``. Exactly one of the two sits on every binding, so the
  direction of an address is a property of the graph rather than a guess from
  its name.
* **Classes** — the device ontology, linked by ``SUBCLASSOF``. A device is
  typed by ``TYPE``. This is what lets "every magnet" find an ``HCorrector``
  without the query naming ``HCorrector``. Classes also carry the synonyms an
  operator would say out loud, as ``skos:altLabel``.

The descriptions are what makes the graph reachable from a phrase rather than
only from a name: "the magnets that bend the beam" matches text no address
spells. They sit on bindings rather than on signals because an address is
ring-qualified and a signal is not — ``SR:MAG:QF:01:CURRENT:SP`` and its
booster counterpart share one signal node but are described differently.

The descriptions and ``system`` come from the generator, so they are there in a
corpus ``build-ttl`` produced — the demo machine — and not in the shipped ALS
corpus, which OSPREY carries as a fixed artifact with no source database to
regenerate from. On that one the way in is a name, an ``altLabel``, a section
or a class.

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
  instead of letting the last one win. Match it with
  ``ANY(l IN c.altLabel WHERE l = $label)``. Comparing it to a string —
  ``c.altLabel = $label`` — is a list against a text and never matches, and the
  string operators do not apply to it at all.


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

In a deployed project the agent rarely needs the first two: whichever verb
seeds or re-verifies the store — the staging step inside every ``osprey up``,
or ``osprey knowledge seed-graph`` — captures the live store's schema (property
lists complete, not sampled) together with the curated examples, and bakes them
into the rendered ``facility-knowledge-graph`` agent prompt, stamped with the
seed marker's checksum. The agent starts already knowing the vocabulary and
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
