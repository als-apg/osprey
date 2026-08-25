.. _how-to-facility-graph:

=========================
Search the Facility Graph
=========================

The facility graph holds the *structure* of the machine: the devices, the
sections they sit in, the classes they belong to, and the control system
addresses bound to each one. It is served by the ``graphdb`` store
(:doc:`../deploy-project/index`) and searched by the **facility-knowledge-graph agent**
— a specialist the main OSPREY agent delegates structural questions to — which
writes Cypher (Neo4j's query language, the graph counterpart of SQL) through
the ``graph`` MCP server. You never write it yourself, and neither does the
main agent: ask it a structural question and the delegation is its own move.

It answers a different kind of question than the :doc:`channel finder
<../use-channel-finder>`. The channel finder turns a phrase into an address —
"beam current" into ``SR:DIAG:DCCT:01:CURRENT:RB``. The graph relates addresses
to each other: how many correctors the storage ring has, what sits in a section
in beam order, which PVs one device exposes and which of them are written, or
which devices share a PV.

.. admonition:: Two things are called "graph"
   :class: important

   This page is about the ``graph`` MCP server — the exploration surface the
   **facility-knowledge-graph** subagent uses to answer the main agent's
   structural questions. The **graph paradigm** is a different thing wearing
   the same word: a channel finder mode (``channel_finder.pipeline_mode:
   graph``) that puts the *channel finder* subagent on this same store to turn
   a phrase into an address, under the ``channel-finder`` server name and with
   a query catalogue written for that job. Same store, two specialist
   subagents, two jobs — see :doc:`../use-channel-finder`.

.. dropdown:: What You'll Learn
   :color: primary
   :icon: book

   - Why the server can only read, and what it refuses before dialing
   - What each degraded state means and the remedy it names
   - How to generate a graph corpus from your own channel databases
   - How to point the tools at a store the facility already runs

   **Prerequisites:** a project with a ``services.graphdb`` block — the
   ``control-assistant`` preset ships one, already seeded.


What the graph holds, how its names are spelled, and what each of the four
tools returns are in :doc:`/reference/contracts/facility-graph`.


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
port settings documented in :doc:`../deploy-project/index`.

.. note::

   ``osprey health`` reports the same store from the outside, under its
   ``graphdb`` category — bolt reachability and how many resources the graph
   holds. See :doc:`../health-and-monitoring/configure-health-checks`.


Leaving the Graph Tools Out
===========================

The server renders wherever ``services.graphdb`` is configured. To keep the
store but take the tools away from the agent — to shrink the tool surface, or
while benchmarking something else — turn the server off explicitly. The
facility-knowledge-graph agent rides the server, so switching it off removes
the subagent and its roster entry in the same build:

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
project's own channel databases, so the graph describes the same machine the
channel finder does. Inside a rendered project:

.. code-block:: console

   $ osprey knowledge build-ttl data/demo_machine.ttl \
       --channel-db data/channel_databases/hierarchical.json \
       --descriptions <your in-context database>
   Wrote data/demo_machine.ttl.
     512 devices, 2908 channel bindings, 113 signals.
     direction from channel limits: data/channel_limits.json
     Load it with: osprey knowledge seed-graph data/demo_machine.ttl

The shipped demo corpus is regenerated from
``src/osprey/templates/apps/control_assistant/`` in the OSPREY source tree
instead, where the two databases sit side by side, so ``--descriptions`` can be
left to its default — a rendered project ships no such neighbour, which is why
the first example names both:

.. code-block:: console

   $ osprey knowledge build-ttl data/demo_machine.ttl \
       --channel-db data/channel_databases/tiers/tier3/hierarchical.json

Every flag, what it decides and what it falls back to — the two databases the
corpus is built from, the file that tells a readback from a setpoint, the
device-family table, and the key to point at the corpus you wrote — is in
:ref:`osprey knowledge <cli-osprey-knowledge>`.

Then load it (see :doc:`okf-bundle` for the seeding verb in full):

.. code-block:: console

   $ osprey knowledge seed-graph data/demo_machine.ttl


Authoring the Ontology
----------------------

The FAMILY-to-class table ``--ontology`` reads is compiled, not written by
hand. What you author is a **LinkML schema** — a YAML file that names each kind
of device on the machine, says which broader kind it belongs to, and lists the
words somebody might use for it when asking. OSPREY's own is
``src/osprey/services/facility_knowledge/ttl_generator/demo_ontology.yaml``;
a facility's looks the same. Two excerpts from it, a class and two families:

.. code-block:: yaml

   classes:

     AcceleratingCavity:
       class_uri: narad_sem:AcceleratingCavity
       is_a: RadioFrequency
       aliases:
         - accelerating cavity
         - cavity
         - rf cavity

   enums:

     DeviceFamily:
       permissible_values:

         BPM:
           meaning: narad_sem:BeamPositionMonitor

         CAVITY:
           meaning: narad_sem:AcceleratingCavity

The class block declares one kind of device: ``is_a`` names its parent, and
``aliases`` are the synonyms it should also answer to — they are emitted as
``skos:altLabel``, which is how the OSPREY agent finds cavities when an
operator writes *rf cavity*. The ``DeviceFamily`` enum is the family map: one
entry per FAMILY token of the channel database, spelled exactly as the database
spells it (hyphens and all, so ``ION-PUMP`` is a valid entry name), with
``meaning`` naming the class devices of that family are typed as.

Compile the schema into the table (this verb needs the ``knowledge`` extra,
``pip install "osprey-framework[knowledge]"``):

.. code-block:: console

   $ osprey knowledge compile-ontology demo_ontology.yaml demo_ontology.json

Then emit the corpus against it with ``build-ttl --ontology
demo_ontology.json``. ``compile-ontology`` also has a ``--check`` form, which
compares an existing table against a fresh compile and fails when the two
have drifted apart:

.. code-block:: console

   $ osprey knowledge compile-ontology demo_ontology.yaml demo_ontology.json --check

``--check`` writes nothing at all — it is the form for CI and for a pre-commit
hook, where the job is to prove a committed table still matches the schema it
came from.

The table can hold less than LinkML can express, so the compiler accepts a
deliberately narrow shape:

* **One root class, named ``AcceleratorDevice``.** Exactly one class leaves
  ``is_a`` unset, and it must be ``AcceleratorDevice`` — the class the NARAD
  vocabulary is rooted at, which is what lets a rollup query written for one
  machine's graph work on another. Every other class descends from it.
* **One parent each.** ``is_a`` names a single parent; there is no multiple
  inheritance.
* **Names in one namespace.** Every class declares
  ``class_uri: narad_sem:<ClassName>``, matching its own name.
* **The enum is the family map.** A schema without a ``DeviceFamily`` enum, or
  with a value that has no ``meaning``, has nothing to compile.

Anything outside that shape is refused when you compile, rather than producing
a table that breaks later, and the message names the element at fault — a class
using ``mixins``, for instance, is reported as *class 'X' uses 'mixins', which
the ontology table cannot represent*.


What the Presets Seed
=====================

Both presets seed ``./data/demo_machine.ttl`` — the demo machine, generated
from the control-assistant preset's own channel databases with the command
above and regenerable the same way. The ARIEL standalone preset ships the same
corpus so that every OSPREY demo describes one facility. Point ``ttl_path`` at
a corpus of your own to seed that instead.

.. note::

   A build profile that ships its own ``data:`` tree replaces the bundle's
   ``data/`` directory wholesale, which takes ``demo_machine.ttl`` with it.
   Such a profile has to set ``services.graphdb.ttl_path`` in its ``config:``
   overlay, at its own corpus. Otherwise the deploy warns, the store comes up
   empty, and every query reports the empty state above.


.. _graph-external-store:

Pointing at a Store the Facility Runs
=====================================

The store does not have to be one this deployment brings up. Give
``services.graphdb`` an explicit ``uri`` and leave ``graphdb`` out of
``deployed_services``:

.. code-block:: yaml

   services:
     graphdb:
       uri: bolt://graph.facility.example:7687
       username: osprey
     # ... the rest of your services, unchanged

   deployed_services: [postgresql, openobserve, qmd]   # no graphdb

Leaving ``graphdb`` in ``deployed_services`` is the mistake to avoid: the build
still stands up a local Neo4j that nobody then queries. ``deployed_services``
is a list, so an override replaces it whole — write out the services you *do*
want rather than the one you are taking away.

Put that account's password in the project ``.env`` as ``GRAPHDB_PASSWORD``.
Nothing mints one here — the store belongs to somebody else, so OSPREY starts
nothing and bootstraps nothing, and it seeds nothing on its own: the corpus is
whatever the facility loaded, unless you run ``osprey knowledge build-ttl`` and
``osprey knowledge seed-graph`` against it deliberately. An external store that
holds no corpus answers every query with zero rows. ``username`` is read only
on this path; a store this deployment does run authenticates as ``neo4j``, the
account its container is created with.

Everything else on this page applies to an external store as written, because
it is all client side: the same four tools, the same read-only posture, the
same bounds and the same degraded states. The graph channel finder paradigm
reads the same block and reaches an external store the same way
(:doc:`../build-profiles`).


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
   * - Channel finder — hierarchical, middle-layer and graph builds
     - 2,908
     - The same set, address for address. A graph-mode build keeps no database
       of its own: it answers out of this corpus.
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
(:doc:`../web-terminal/multi-user/index`) get the same facility-knowledge-graph agent and its tools,
reading the hosting deployment's store. They are attached renders — they deploy no services of their own — so
they have to be told which port that store is published on. Per-user web
terminal containers run with ``network_mode: host``, so a container's
``localhost`` *is* the deployment host, and both presets pin the bolt port
directly:

.. code-block:: yaml

   config:
     services.graphdb.port_host: 7687

Two consequences worth knowing before you move anything:

* **Move the port on the hosting deployment and every persona follows.** A
  persona built beside its deployment is told the store's bolt port from the
  deployment's own render — the same way it is told the qmd sidecar's port,
  the Postgres the logbook lives in and the rest (:doc:`/how-to/build-profiles`) — so
  nothing in a persona preset restates it, and a persona that restates it
  with a different number is refused at build time.
* A ``deployment.bind_address`` pinned to a specific non-loopback interface
  publishes the store there and not on ``localhost``, so the personas would no
  longer reach it. URL-backed panels have the same shape — they name
  ``http://127.0.0.1:<port>`` outright (:doc:`../web-terminal/panels`) — so a
  deployment that moves off loopback has to revisit both.

The read-only persona receives ``GRAPHDB_PASSWORD`` as well. The store has a
single write-capable account, so read-only-ness here is enforced by the graph
server's read transaction rather than by the credential — the same posture the
ARIEL database credential already takes. ``control-assistant-ariel`` has no
control surface and no graph tools by design.


.. seealso::

   :doc:`/reference/contracts/facility-graph`
      What the graph holds, the Cypher spelling rules, and the four tools.

   :doc:`../deploy-project/index`
      The ``services.graphdb`` block: image, ports, corpus, memory, and the
      query bounds.

   :ref:`osprey knowledge <cli-osprey-knowledge>`
      Every verb in the command group, with its flags and defaults.

   :doc:`okf-bundle`
      The bundle of concept documents ``seed-from-ttl`` writes, and what
      ``seed-graph`` does to a store that already holds a corpus.

   :doc:`../use-channel-finder`
      The other way into the same machine — phrases to addresses.

   :doc:`/architecture/mcp-servers`
      MCP servers provided by the framework, including ``graph``.
