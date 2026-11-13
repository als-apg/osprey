.. _architecture-facility-relations:

Facility Relations
==================

A relation says how two parts of the machine depend on each other: a magnet is
powered by a supply, a supply sits in a rack, a monitor is read out by a
chassis. OSPREY records each such statement as a **claim** in the
:doc:`facility file <facility-file>`, never as a field on a device. This page
explains that choice and follows a claim from its record to the views that
read it. The record format is in :doc:`/how-to/describe-your-facility`; what
the graph writes for a claim is in :ref:`reference-facility-graph`.

.. raw:: html
   :file: ../_diagrams/facility-relations.html

Claims, not slots
=================

A slot such as ``powered_by`` on a device holds one value. Two sources that
disagree about it are two layers writing one field, and the build stops with
``layer-conflict``: the facility cannot hold the disagreement it actually has.
A slot also has nowhere to put the port, the cable number or the document the
value came from, and a place or a group cannot fill one.

A claim is one record from one source, with an id of its own. It names a
relation type, a subject and an object (each a device, a place or a group,
with an optional port or unit), a ``basis`` (``stated``, or ``inferred`` with
the ``rule`` that inferred it), typed attributes, and a ``source`` citing a
document, its version, its state and a locator. Two claims never merge, so two
sources that disagree build as two claims, and the build decides what the
disagreement means.

The chain
=========

**Records.** ``records/relations.yaml`` holds the claims. ``documents.yaml``
declares the documents they cite, each with a ``kind`` (``rack-layout``,
``cable-schedule``, ...). ``policy.yaml`` is optional; its
``relations.precedence`` ranks document kinds per relation type or by
``default``, and ``stated_over_inferred`` (true unless set false) ranks a
stated claim before an inferred one.

**Standing.** A claim whose shape breaks a rule of its type stops the build
with ``relation-invalid``. Every sound claim gets a ``standing``, decided per
key: what the type allows one object for, such as the subject of
``located_in`` or the subject and port of ``powered_by``. Claims on one key
group by the object they state. One group is ``accepted``. Several groups without a
policy are ``contested``. With a policy they rank by their document's kind,
then by basis; the strictly best group is ``accepted``, the rest
``superseded``, and a tie at the best stays ``contested``. A claim on several
keys keeps its worst standing. A claim that states
less than the facility knows (a place end where the type allows a device, or a
key attribute, port or unit left out) is coarse: accepted, but no part of the
ranking. Standing raises warnings and the build goes on:
``relation-contested`` once per unsettled key, ``relation-coarse`` once per
type.

**Projection.** Accepted ``located_in`` claims fill ``Device.place`` when the
device has no place yet, no claim on its key is contested, and the accepted
claims that are not coarse name exactly one place. The device then carries
``provenance.place_from: relation``; the claim's unit is not copied. A disputed
key projects nothing, and its warning is the report.

**Views.** The graph view reads every standing: an accepted claim is an edge,
a contested claim is an edge under the contested twin of its predicate, a
superseded claim is no edge, and the ``facility_graph`` settings choose which
claims also become claim nodes. The facts view and the channel finder's
hierarchical and middle-layer indexes read the projected place, not the
claims. The in-context index, pyAML, simulator, Bluesky, limits and benchmark
views read no relation; a claim changes none of them.

The sector-3 example
====================

The example facility ``osprey init`` ships carries one worked example around
the storage-ring quadrupole ``SR/QF05`` and the monitor ``SR/BPM13``: four
documents (an equipment list, a rack layout, a cable schedule and a network
plan), eleven devices, the rack row ``HALL/ROW3`` with racks ``RK07`` and
``RK08``, the vacuum-section group ``SR/VS03``, and 18 claims, 16 stated and 2
inferred. It ships no ``policy.yaml``. Built as it ships, 16 claims are
accepted and 2 contested, and ``osprey facility validate`` prints two warnings
and exits 0.

**The contested rack.** ``qf05-ps-rack-a`` puts the supply ``SR/QF05-PS`` in
``HALL/ROW3/RK07``, citing the released rack layout; ``qf05-ps-rack-b`` puts
it in ``HALL/ROW3/RK08``, citing the in-work cable schedule. Neither document
ranks first, so both claims are contested, the build warns
``relation-contested``, the graph holds two ``LOCATEDINCONTESTED`` edges, and
``SR/QF05-PS`` stays unplaced.

**The coarse cable.** ``w-03-0240`` cables port ``A`` of ``SR/BPM13`` to the
rack ``HALL/ROW3/RK08`` rather than to a device in it. The claim is accepted
and the build warns ``relation-coarse`` for ``cabled_to``.

**The inferred readout.** ``bpm13-readout`` says ``SR/BPM13`` is read out by
channel ``CH1`` of the chassis ``SR/BPM13-EL``. No document states it; the
rule ``bpm-electronics-in-cable-destination-rack`` inferred it from the cable
schedule row of the coarse cable. The chassis itself takes its place
``HALL/ROW3/RK08`` from its uncontested rack claim ``bpm13-el-rack``.

**The undecided identity.** ``qf05-identity`` is an inferred ``same_as``
between ``SR/QF05`` and ``SR/MAG-Q-0301``, by the rule
``family-and-ordinal-match``. ``same_as`` records an identity nobody has
decided; it merges nothing. ``SR/MAG-Q-0301`` counts as a quadrupole of its
own in the facts view, the graph and the demo's device counts.

The group ``SR/VS03`` carries no description, though the design example the
sector is taken from gives it one. The graph takes a device's family description from the smallest
described group that names it, so a described three-member group would
replace the family descriptions of ``SR/QF05``, ``SR/BPM13`` and
``SR/ION-PUMP13``, and the graph's purpose search would stop finding them.

**Adding the precedence policy.** A ``policy.yaml`` with
``precedence: {located_in: [rack-layout, cable-schedule]}`` ranks the rack
layout first. ``qf05-ps-rack-a`` is then accepted and ``qf05-ps-rack-b``
superseded: 17 claims accepted, 1 superseded. ``SR/QF05-PS`` takes the place
``HALL/ROW3/RK07`` with ``place_from: relation``, the contested warning goes,
and only the coarse warning remains. In the graph the two contested edges
become one ``LOCATEDIN``, and the superseded claim is no edge.

Divergence from NARAD
=====================

The NARAD seed schema's device record declares four relation slots:
``powered_by``, ``housed_by``, ``controller`` and ``local_system``. In the seed
they are optional and untyped: no target kind, no cardinality, no attributes,
no source. None is carried into ``core.yaml``, and since none is required they
are not in ``loosenings.yaml`` either. Each is a relation type instead:
``powered_by`` is ``powered_by``, ``housed_by`` is ``located_in`` for a place
or ``part_of`` for a chassis, and ``controller`` is ``controlled_by``.
``local_system`` maps to no type; where a device sits is its place.
