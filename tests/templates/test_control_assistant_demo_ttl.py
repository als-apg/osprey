"""The demo machine corpus the control-assistant data tree ships.

A committed corpus that nobody counts can drift silently: a change to it would
quietly ship a different graph. The census below (512 devices, 2908 bindings,
396 written and 2512 read signals) is the demo machine as of the corpus
committed beside this test, and the 396 writes are exactly its `:SP`
addresses. The prose census sits beside it: three
description predicates on every binding, three more plus the SYSTEM token on
every device, and none of the six on a semantic signal.

The uppercase check guards the other half of the pipeline. neosemantics imports
a predicate IRI under the local name it finds, so `narad_p:hasBinding` becomes
the relationship type `hasBinding` — but n10s also has a `LABELS_AND_NODES`
handling that would uppercase it to `HASBINDING`, and the example queries the
agent is given all spell the camelCase form. A corpus carrying uppercase names
would import into a graph where every shipped query returns nothing.
"""

from pathlib import Path

import pytest

#: The control-assistant preset's data tree, where the corpus ships.
DEMO_DATA = Path(__file__).resolve().parents[2] / "src/osprey/templates/apps/control_assistant/data"

#: The corpus as it ships in the preset's data tree.
TEMPLATE_TTL = DEMO_DATA / "demo_machine.ttl"

#: NARAD property namespace — the one the emitter binds as ``narad_p:``.
NARAD_PROPERTY = "https://narad.example.org/property/"

#: NARAD shared-semantics namespace — ``narad_sem:``, where the device classes
#: and the two structural classes live.
NARAD_SEMANTICS = "https://narad.example.org/schema/shared_semantics/"

#: Classes in ``narad_sem:`` that type something other than a device.
NON_DEVICE_CLASSES = {"ChannelBinding", "SemanticSignal"}

#: The demo machine's census, as generated from the tier-3 channel database and
#: the shipped channel limits.
EXPECTED_DEVICES = 512
EXPECTED_BINDINGS = 2908
EXPECTED_WRITES = 396
EXPECTED_READS = 2512

#: Prose predicates carried once by every ``narad_sem:ChannelBinding``: the
#: channel's own sentence and the text of the two address tokens it ends in.
BINDING_DESCRIPTION_PREDICATES = ("description", "fieldDescription", "subfieldDescription")

#: Prose predicates carried once by every device node, from the tree levels
#: above it, plus the SYSTEM token itself — the one address token the device
#: IRI does not spell, so it ships as data a query can filter on.
DEVICE_DESCRIPTION_PREDICATES = ("familyDescription", "systemDescription", "ringDescription")
DEVICE_TOKEN_PREDICATES = ("system",)

#: n10s' uppercase spellings of the three relationship types the shipped
#: example queries use. Any of these in the corpus means the queries miss.
UPPERCASE_N10S_NAMES = ("HASBINDING", "READSSIGNAL", "WRITESSIGNAL")


@pytest.fixture(scope="module")
def graph() -> object:
    """The committed corpus, parsed once."""
    from rdflib import Graph

    parsed = Graph()
    parsed.parse(TEMPLATE_TTL, format="turtle")
    return parsed


# ---------------------------------------------------------------------------
# What the corpus contains
# ---------------------------------------------------------------------------


def test_binding_and_signal_census(graph) -> None:
    from rdflib import URIRef

    def count(local_name: str) -> int:
        return len(list(graph.triples((None, URIRef(NARAD_PROPERTY + local_name), None))))

    assert count("hasBinding") == EXPECTED_BINDINGS
    assert count("writesSignal") == EXPECTED_WRITES
    assert count("readsSignal") == EXPECTED_READS


def test_prose_census(graph) -> None:
    """Every binding and every device carries its prose, exactly once.

    The subject counts are what make this a census rather than a spot check: a
    predicate appearing 2,908 times spread over 40 bindings would satisfy a
    triple count and import into a graph where most channels have no text at
    all. neosemantics keeps one value per property unless told otherwise, so a
    doubled predicate is also silent data loss at seed time.
    """
    from rdflib import URIRef

    def subjects(local_name: str) -> set:
        return set(graph.subjects(URIRef(NARAD_PROPERTY + local_name), None))

    def triples(local_name: str) -> int:
        return len(list(graph.triples((None, URIRef(NARAD_PROPERTY + local_name), None))))

    for predicate in BINDING_DESCRIPTION_PREDICATES:
        assert triples(predicate) == EXPECTED_BINDINGS, (
            f"narad_p:{predicate} appears {triples(predicate)} times, not once per binding."
        )
        assert len(subjects(predicate)) == EXPECTED_BINDINGS

    for predicate in DEVICE_DESCRIPTION_PREDICATES + DEVICE_TOKEN_PREDICATES:
        assert triples(predicate) == EXPECTED_DEVICES, (
            f"narad_p:{predicate} appears {triples(predicate)} times, not once per device."
        )
        assert len(subjects(predicate)) == EXPECTED_DEVICES


def test_semantic_signals_carry_no_description(graph) -> None:
    """No prose predicate lands on a ``narad_sem:SemanticSignal``.

    A signal is keyed without a ring, and the tree's field and subfield prose is
    written per ring — so text on a signal could only be one ring's wording
    standing in for every ring's. The corpus puts that text on bindings, whose
    address carries all six tokens.
    """
    from rdflib import RDF, URIRef

    signals = set(graph.subjects(RDF.type, URIRef(NARAD_SEMANTICS + "SemanticSignal")))
    assert signals, "The corpus declares no semantic signals at all"

    for predicate in BINDING_DESCRIPTION_PREDICATES + DEVICE_DESCRIPTION_PREDICATES:
        described = signals & set(graph.subjects(URIRef(NARAD_PROPERTY + predicate), None))
        assert not described, f"narad_p:{predicate} is on {len(described)} semantic signals."


def test_every_device_carries_a_semantic_class(graph) -> None:
    from rdflib import RDF

    devices = {
        subject
        for subject, cls in graph.subject_objects(RDF.type)
        if str(cls).startswith(NARAD_SEMANTICS)
        and str(cls).removeprefix(NARAD_SEMANTICS) not in NON_DEVICE_CLASSES
    }
    assert len(devices) == EXPECTED_DEVICES


def test_only_setpoints_are_written(graph) -> None:
    """Every written binding is a ``:SP`` address, and only those are written."""
    from rdflib import URIRef

    def pvs(local_name: str) -> set[str]:
        full_pv = URIRef(NARAD_PROPERTY + "fullPv")
        return {
            str(pv)
            for binding in graph.subjects(URIRef(NARAD_PROPERTY + local_name), None)
            for pv in graph.objects(binding, full_pv)
        }

    written = pvs("writesSignal")
    assert len(written) == EXPECTED_WRITES
    assert all(pv.endswith(":SP") for pv in written)
    assert not any(pv.endswith(":SP") for pv in pvs("readsSignal"))


def test_no_uppercase_n10s_names() -> None:
    text = TEMPLATE_TTL.read_text(encoding="utf-8")
    for name in UPPERCASE_N10S_NAMES:
        assert name not in text
