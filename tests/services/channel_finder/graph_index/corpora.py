"""Small Turtle corpora exercising the corners of the index builder's rules.

Each constant is a complete corpus (prefixes included) written so that a test
can state its expected rows by hand. The names say which rule a corpus probes;
the parity lane seeds the same strings into an n10s store, so a corpus here must
stay valid NARAD Turtle rather than a builder-only shorthand.

The URIs are spelled in the demo corpus's namespaces so the rows read like the
ones the shipped index carries.
"""

from __future__ import annotations

PREFIXES = """\
@prefix narad_p: <https://narad.example.org/property/> .
@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

NARAD_SEM = "https://narad.example.org/schema/shared_semantics/"
DEVICE = "https://narad.example.org/device/"
BINDING = "https://narad.example.org/binding/"

#: The class tree, signal and binding types every corpus below shares. The
#: chain ``Quadrupole ⊂ Magnet ⊂ AcceleratorDevice ⊂ owl:Thing`` is the demo
#: corpus's shape; ``owl:Thing`` is deliberately *not* declared an
#: ``owl:Class``, so it must fall out of every ancestor and parent list, and
#: ``Sextupole`` is a device class nothing is typed as, which pruning drops.
SHARED_ONTOLOGY = """
narad_sem:ChannelBinding a owl:Class .
narad_sem:SemanticSignal a owl:Class .

narad_sem:AcceleratorDevice a owl:Class ;
    rdfs:subClassOf owl:Thing .

narad_sem:Magnet a owl:Class ;
    rdfs:subClassOf narad_sem:AcceleratorDevice ;
    skos:altLabel "magnet" .

narad_sem:Quadrupole a owl:Class ;
    rdfs:subClassOf narad_sem:Magnet ;
    skos:altLabel "quad",
        "quadrupole",
        "Focusing Magnet" .

narad_sem:Sextupole a owl:Class ;
    rdfs:subClassOf narad_sem:Magnet ;
    skos:altLabel "sext" .

narad_sem:quad_current_sp a narad_sem:SemanticSignal ;
    rdfs:label "quad_current_sp" .

narad_sem:quad_current_rb a narad_sem:SemanticSignal ;
    rdfs:label "quad_current_rb" .
"""

#: One quadrupole with a write binding, a read binding and a binding with no
#: signal edge. Probes the subclass chain, ``altLabel`` folding into the
#: haystack, the ``R``/``W``/no-edge shapes, class-row counts and pruning.
SUBCLASS_CHAIN = (
    PREFIXES
    + SHARED_ONTOLOGY
    + """
<https://narad.example.org/device/demo_SR_QF1> a narad_sem:Quadrupole ;
    narad_p:hasBinding <https://narad.example.org/binding/QF1_SP>,
        <https://narad.example.org/binding/QF1_RB>,
        <https://narad.example.org/binding/QF1_NOTE> ;
    narad_p:sectionCode "SR" ;
    narad_p:sourceName "QF1" ;
    narad_p:system "MAG" .

<https://narad.example.org/binding/QF1_SP> a narad_sem:ChannelBinding ;
    narad_p:bindingId "narad:binding:demo:SR:QF1:Setpoint" ;
    narad_p:description "Quadrupole QF1 current setpoint" ;
    narad_p:fullPv "SR:MAG:QF1:CURRENT:SP" ;
    narad_p:writesSignal narad_sem:quad_current_sp .

<https://narad.example.org/binding/QF1_RB> a narad_sem:ChannelBinding ;
    narad_p:bindingId "narad:binding:demo:SR:QF1:Monitor" ;
    narad_p:description "Quadrupole QF1 current readback" ;
    narad_p:fullPv "SR:MAG:QF1:CURRENT:RB" ;
    narad_p:readsSignal narad_sem:quad_current_rb .

<https://narad.example.org/binding/QF1_NOTE> a narad_sem:ChannelBinding ;
    narad_p:fullPv "SR:MAG:QF1:NOTE" .
"""
)

#: A binding carrying both edges (to two signals), and one reading a signal
#: twice through both predicates (one signal entry, two edges). ``RW`` is the
#: direction both shapes answer.
BOTH_EDGES = (
    PREFIXES
    + SHARED_ONTOLOGY
    + """
<https://narad.example.org/device/demo_SR_QF2> a narad_sem:Quadrupole ;
    narad_p:hasBinding <https://narad.example.org/binding/QF2_BOTH>,
        <https://narad.example.org/binding/QF2_SAME> ;
    narad_p:sectionCode "SR" ;
    narad_p:sourceName "QF2" ;
    narad_p:system "MAG" .

<https://narad.example.org/binding/QF2_BOTH> a narad_sem:ChannelBinding ;
    narad_p:fullPv "SR:MAG:QF2:CURRENT" ;
    narad_p:readsSignal narad_sem:quad_current_rb ;
    narad_p:writesSignal narad_sem:quad_current_sp .

<https://narad.example.org/binding/QF2_SAME> a narad_sem:ChannelBinding ;
    narad_p:fullPv "SR:MAG:QF2:SAME" ;
    narad_p:readsSignal narad_sem:quad_current_sp ;
    narad_p:writesSignal narad_sem:quad_current_sp .
"""
)

#: Two devices each binding their own ``ChannelBinding`` node under one
#: ``fullPv``, one reading and one writing. The store answers two search rows;
#: the roster collapses them to one address with no direction.
SHARED_FULL_PV = (
    PREFIXES
    + SHARED_ONTOLOGY
    + """
<https://narad.example.org/device/demo_SR_QF3> a narad_sem:Quadrupole ;
    narad_p:hasBinding <https://narad.example.org/binding/QF3_SHARED> ;
    narad_p:sectionCode "SR" ;
    narad_p:sourceName "QF3" ;
    narad_p:system "MAG" .

<https://narad.example.org/device/demo_SR_QF4> a narad_sem:Quadrupole ;
    narad_p:hasBinding <https://narad.example.org/binding/QF4_SHARED> ;
    narad_p:sectionCode "SR" ;
    narad_p:sourceName "QF4" ;
    narad_p:system "MAG" .

<https://narad.example.org/binding/QF3_SHARED> a narad_sem:ChannelBinding ;
    narad_p:fullPv "SR:MAG:SHARED:CURRENT" ;
    narad_p:readsSignal narad_sem:quad_current_rb .

<https://narad.example.org/binding/QF4_SHARED> a narad_sem:ChannelBinding ;
    narad_p:fullPv "SR:MAG:SHARED:CURRENT" ;
    narad_p:writesSignal narad_sem:quad_current_sp .
"""
)

#: One binding node hung under two devices. ``HASBINDING`` is matched per
#: device, so the store answers it twice, once with each device's columns.
BINDING_UNDER_TWO_DEVICES = (
    PREFIXES
    + SHARED_ONTOLOGY
    + """
<https://narad.example.org/device/demo_SR_QF5> a narad_sem:Quadrupole ;
    narad_p:hasBinding <https://narad.example.org/binding/TWICE> ;
    narad_p:sectionCode "SR" ;
    narad_p:sourceName "QF5" ;
    narad_p:system "MAG" .

<https://narad.example.org/device/demo_BR_QF6> a narad_sem:Quadrupole ;
    narad_p:hasBinding <https://narad.example.org/binding/TWICE> ;
    narad_p:sectionCode "BR" ;
    narad_p:sourceName "QF6" ;
    narad_p:system "MAG" .

<https://narad.example.org/binding/TWICE> a narad_sem:ChannelBinding ;
    narad_p:fullPv "SR:MAG:TWICE:CURRENT" ;
    narad_p:readsSignal narad_sem:quad_current_rb .
"""
)

#: A device placed nowhere: no ``sectionCode``, no ``system``, and no
#: ``sourceName`` either. Its columns are NULL, never the empty string, and
#: its haystack carries only what the corpus states.
DEVICE_WITHOUT_SECTION_OR_SYSTEM = (
    PREFIXES
    + SHARED_ONTOLOGY
    + """
<https://narad.example.org/device/demo_NOWHERE> a narad_sem:Quadrupole ;
    narad_p:hasBinding <https://narad.example.org/binding/NOWHERE_RB> .

<https://narad.example.org/binding/NOWHERE_RB> a narad_sem:ChannelBinding ;
    narad_p:description "Unplaced readback" ;
    narad_p:fullPv "NOWHERE:RB" ;
    narad_p:readsSignal narad_sem:quad_current_rb .
"""
)

#: ``Loop_A ⊂ Loop_B ⊂ Loop_A``, a class naming itself as its own parent, and
#: a device typed ``Loop_A``. A bounded walk terminates, both loop classes
#: roll the device up, and the self-parent survives as a parent entry (the
#: pruning already knows to ignore it).
CLASS_CYCLE = (
    PREFIXES
    + """
narad_sem:ChannelBinding a owl:Class .
narad_sem:SemanticSignal a owl:Class .

narad_sem:Loop_A a owl:Class ;
    rdfs:subClassOf narad_sem:Loop_B .

narad_sem:Loop_B a owl:Class ;
    rdfs:subClassOf narad_sem:Loop_A .

narad_sem:Selfish a owl:Class ;
    rdfs:subClassOf narad_sem:Selfish .

narad_sem:loop_signal a narad_sem:SemanticSignal .

<https://narad.example.org/device/demo_LOOP> a narad_sem:Loop_A ;
    narad_p:hasBinding <https://narad.example.org/binding/LOOP_RB> ;
    narad_p:sourceName "LOOP" .

<https://narad.example.org/device/demo_SELF> a narad_sem:Selfish ;
    narad_p:hasBinding <https://narad.example.org/binding/SELF_RB> ;
    narad_p:sourceName "SELF" .

<https://narad.example.org/binding/LOOP_RB> a narad_sem:ChannelBinding ;
    narad_p:fullPv "LOOP:RB" ;
    narad_p:readsSignal narad_sem:loop_signal .

<https://narad.example.org/binding/SELF_RB> a narad_sem:ChannelBinding ;
    narad_p:fullPv "SELF:RB" .
"""
)

#: A chain twelve classes deep, ``Deep_00 ⊂ Deep_01 ⊂ … ⊂ Deep_11``, with a
#: device typed ``Deep_00``. The store's ``*0..10`` reaches ``Deep_10`` and not
#: ``Deep_11``.
DEEP_CHAIN = (
    PREFIXES
    + """
narad_sem:ChannelBinding a owl:Class .
"""
    + "".join(
        f"narad_sem:Deep_{i:02d} a owl:Class ;\n    rdfs:subClassOf narad_sem:Deep_{i + 1:02d} .\n\n"
        for i in range(11)
    )
    + """
narad_sem:Deep_11 a owl:Class .

<https://narad.example.org/device/demo_DEEP> a narad_sem:Deep_00 ;
    narad_p:hasBinding <https://narad.example.org/binding/DEEP_RB> .

<https://narad.example.org/binding/DEEP_RB> a narad_sem:ChannelBinding ;
    narad_p:fullPv "DEEP:RB" .
"""
)

#: The labels n10s would not put on a node. ``UNTYPED`` is a ``hasBinding``
#: target that is never typed ``ChannelBinding``, so the store's match skips
#: it; ``QF7_RB`` reads a signal node that is never typed ``SemanticSignal``,
#: so the store binds no edge and no signal for it; ``QF7_LABELLESS`` reads a
#: typed signal that carries no ``rdfs:label``, so its name is the URI tail;
#: and the device's second type is not declared a class, so it contributes
#: no ancestor. ``owl:Thing`` gets a ``sectionCode`` to show the section
#: census counts every subject.
UNTYPED_TARGETS = (
    PREFIXES
    + SHARED_ONTOLOGY
    + """
narad_sem:unlabelled_signal a narad_sem:SemanticSignal .

<https://narad.example.org/device/demo_SR_QF7> a narad_sem:Quadrupole, narad_sem:NotAClass ;
    narad_p:hasBinding <https://narad.example.org/binding/QF7_RB>,
        <https://narad.example.org/binding/QF7_LABELLESS>,
        <https://narad.example.org/binding/UNTYPED> ;
    narad_p:sectionCode "SR" ;
    narad_p:sourceName "QF7" ;
    narad_p:system "MAG" .

<https://narad.example.org/binding/QF7_RB> a narad_sem:ChannelBinding ;
    narad_p:fullPv "SR:MAG:QF7:RB" ;
    narad_p:readsSignal narad_sem:not_a_signal .

<https://narad.example.org/binding/QF7_LABELLESS> a narad_sem:ChannelBinding ;
    narad_p:fullPv "SR:MAG:QF7:LABELLESS" ;
    narad_p:readsSignal narad_sem:unlabelled_signal .

<https://narad.example.org/binding/UNTYPED>
    narad_p:fullPv "SR:MAG:QF7:UNTYPED" ;
    narad_p:readsSignal narad_sem:quad_current_rb .

owl:Thing narad_p:sectionCode "LTB" .
"""
)

#: Valid Turtle that binds nothing: the ontology alone. Not an error.
NO_BINDINGS = PREFIXES + SHARED_ONTOLOGY

#: Not Turtle at all: an unterminated triple.
INVALID_TURTLE = PREFIXES + "<https://narad.example.org/device/broken> a narad_sem:Quadrupole\n"
