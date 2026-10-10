.. _architecture-facility-file:

===========================
Facility File and Its Views
===========================

OSPREY keeps one description of a facility and derives everything else from
it. The sources under ``data/facility/`` are combined by ``osprey build`` into
the *facility file*, and each service reads a *view* of that file, a file in
the shape the service needs. No service keeps a channel list of its own, so
the channel finder, the limits check, the simulator and the knowledge graph
cannot disagree about which channels exist or what they are.

Writing the sources is in :doc:`/how-to/describe-your-facility`. This page is
about what the build makes of them.

.. code-block:: text

   data/facility/                build/
     identity.yaml                 facility.json
     records/        osprey        data/simulator/
     models.yaml     build         data/channel_limits.json
     limits.yaml    ───────►       data/facility_facts.json, .md
     seeds.yaml                    data/graph/facility.ttl
     fixes.yaml                    data/channel_finder/<mode>.json
     imported/<layer>/             data/bluesky_devices.yml
     ...

The facility file
=================

``build/facility.json`` is every record of the facility in one document:
``identity``, ``classes``, ``places``, ``devices``, ``channels``, ``groups``,
``models`` with their wiring, ``limits`` and ``scenarios``. It holds two
things no source file does. The first is what the build computes: the readback
each setpoint is paired with, each device's place and position along its
model's deck, a device's ordinal among its class, and each wiring record's
unit, direction and operating point. The second is ``provenance``: per record,
which layer and file stated each field, which fixes were applied and which
fields the build filled. ``osprey facility show <id>`` prints it.

The file is the same in every render of a build. It carries no timestamp, no
version and no absolute path, so equal sources give equal bytes.

Two halves of the build
=======================

The first half makes the facility file, once, in memory. It runs in fixed
stages and a stage runs only when every earlier one is clean:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Stage
     - What it does
   * - load
     - Parses each source file and refuses unknown keys and fields only the
       build may write.
   * - combine
     - Merges the layers field by field, applies ``fixes.yaml`` and fills the
       defaults.
   * - schema
     - Holds the combined file to the facility schema.
   * - references
     - Every id a record names exists; device classes and signal roles are
       known.
   * - records
     - The pair, value, limits and seed rules.
   * - compute
     - Spans, places, positions, wiring and engines, read off the decks.

The second half writes. A build renders more than one tree (the deployment's
own, one per persona, one per container image) and each render receives the
same facility file and the views its own configuration asks for. A view a
render does not carry is named on stderr with the configuration key that left
it out; a channel-finder index the profile did not select is neither written
nor named.

``osprey facility validate`` runs both halves into a temporary directory and
writes nothing into the repo.

The views
=========

.. list-table::
   :header-rows: 1
   :widths: 16 30 20 34

   * - View
     - Written to (under the render)
     - Written when
     - Read by
   * - simulator
     - ``data/simulator/``
     - always
     - The Virtual Accelerator container, the same simulator served in
       process and the ``osprey sim`` commands.
   * - limits
     - ``data/channel_limits.json``
     - always
     - The connector's reference monitor and the runtime's limits check, on
       every write.
   * - facts
     - ``data/facility_facts.json``, ``data/facility_facts.md``
     - always
     - The agents' prompts and the ``facility_description`` tool.
   * - bluesky
     - ``data/bluesky_devices.yml``
     - the render runs a Bluesky lane
     - The Bluesky worker, as its device file.
   * - in_context, hierarchical, middle_layer
     - ``data/channel_finder/<mode>.json``; ``middle_layer.duckdb`` beside the
       middle-layer index
     - ``channel_finder.pipeline_mode`` selects it
     - The channel finder.
   * - graph
     - ``data/graph/facility.ttl``
     - always
     - The graph store ``osprey up`` seeds, and ``osprey knowledge
       seed-from-ttl``.

**Simulator.** The view lists every channel address of the facility
(``addresses.json``), every model with its wiring (``variables.json``), the
models this render serves (``served_models.json``), the seeds, the scenarios
and a byte copy of each deck. A simulator serves exactly the channels of
``addresses.json``; an address outside it does not exist.

**Limits.** One entry per record of ``limits.yaml`` and nothing else. A
channel without a record is decided at write time by
``control_system.limits_checking.mode``.

**Facts.** What agents are told about the facility: its identity, the place
tree with device counts, the device classes with their aliases, the signals in
use and the models. It is how an agent knows the facility's own words without
a prompt written by hand.

**Bluesky.** One settable per setpoint, with its paired readback and the
tolerance it settles within, and one readable per readback.

**Channel finder.** Three indexes, one per file-backed pipeline; a render
carries the one its profile's ``channel_finder_mode`` selects. The fourth
pipeline, ``graph``, reads the graph store.

**Graph.** The facility file as Turtle: one node per place, device, channel
and group, with the device classes as a class tree.

.. _architecture-facility-schema:

The schema
==========

The facility file's shape is two LinkML schema modules,
``src/osprey/facility/schema/core.yaml`` (the records) and ``vocabulary.yaml``
(the shared device-class tree, the signal roles and the property names, each
with the aliases people use for them). The build validates the combined file
against the model generated from them.

The two modules carry NARAD's LinkML schemas forward. NARAD requires more of a facility than a bare channel
list can state, so ``loosenings.yaml``, beside the two modules, records what
became of each slot NARAD requires: one row per slot with its ``fate``
(``dropped``, ``optional``, ``renamed:<slot>`` or ``header``) and the reason.
In the other direction the graph view writes the facility file as Turtle in
NARAD's terms, the ``narad_p:`` predicates and ``narad_sem:`` classes of
``data/graph/facility.ttl``, which is what the knowledge graph loads.

A facility extends the vocabulary without touching the schema: a device class
it adds in ``classes.yaml`` is a child of a vocabulary class, and reaches the
facts view and the graph's class tree with its aliases.

.. _architecture-simulation-models:

Simulation models
=================

A facility file's ``models`` are what stands behind the channels when the
deployment runs against a simulator. Each model record names an ``engine``, a
``deck`` and the ``wiring`` that ties channel addresses to elements of the
deck.

**Engines are plug-ins.** An engine is a module registered in the
``osprey.simulation.engines`` entry-point group under the name a model's
``engine`` gives. OSPREY ships ``pyat``. The build reaches the engine to read
the deck (where each element sits, each wired channel's operating point, what
each wiring record is) without building a model, and stops with
``engine-missing`` when no installed package registers the name. The contract
is under :ref:`extending-lume-model`.

**The texture model serves the rest.** Every channel no model wires is served
by ``texture``, from its seed: a setpoint holds the value written to it, a
readback follows its nominal with the noise and drift its seed states.
``texture`` is always served, so a facility with a channel list and no physics
model is still a whole machine to a client.

**A render chooses what it serves.** ``simulation.models`` names the physics
models a render serves; absent, it serves every model the facility file holds,
and ``[]`` serves ``texture`` alone. The choice is written into the simulator
view's ``served_models.json``. Every model's record and deck are in the view
whether or not it is served, so the set can change without a different view
shape.

**A failed model does not take the machine down.** Each served physics model
has a status channel, ``<code>:SIM:<model>:STATUS``, which exists in no other
view. A model whose engine cannot build it, over a deck that does not load for
one, is left failed: its status channel carries the
engine's error text, and every other model and every texture channel is still
served. ``osprey sim status`` prints one ``<model>: <status>`` line per served
model, ``ok`` or that text, and names the log the simulator appends to.

How the composite serves the models over Channel Access and PVAccess is in
:doc:`virtual-accelerator`.

.. seealso::

   :doc:`/how-to/describe-your-facility`
      The source tree, file by file.

   :doc:`/how-to/control-systems/use-virtual-accelerator`
      Running the simulator and serving a facility's own model.

   :doc:`/reference/configuration/config`
      ``simulation.models``, ``simulation.tick_s`` and
      ``control_system.limits_checking.mode``.
