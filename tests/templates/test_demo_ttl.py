"""The graph view the demo build writes.

A corpus that nobody counts can drift silently: a change to the facility or to
the view's writer would quietly seed a different graph. The census below (1016
devices, 4731 bindings, 674 written and 4052 read signals) is the demo facility
as the build renders it, and the 674 writes are all `:SP` addresses.
The prose census sits beside it: a description on every binding, a system
description plus the SYSTEM token on every device, a family description on
every device a described group names, and none of the three descriptions on a
semantic signal.

The uppercase check guards the other half of the pipeline. neosemantics imports
a predicate IRI under the local name it finds, so `narad_p:hasBinding` becomes
the relationship type `hasBinding` — but n10s also has a `LABELS_AND_NODES`
handling that would uppercase it to `HASBINDING`, and the example queries the
agent is given all spell the camelCase form, `partOf` and `locatedIn`
included. A corpus carrying uppercase names
would import into a graph where every shipped query returns nothing.
"""

from pathlib import Path

import pytest
from tests._builds import BuiltProject

#: NARAD property namespace — the one the emitter binds as ``narad_p:``.
NARAD_PROPERTY = "https://narad.example.org/property/"

#: NARAD shared-semantics namespace — ``narad_sem:``, where the device classes
#: and the two structural classes live.
NARAD_SEMANTICS = "https://narad.example.org/schema/shared_semantics/"

#: Classes in ``narad_sem:`` that type something other than a device.
NON_DEVICE_CLASSES = {"ChannelBinding", "SemanticSignal"}

#: The demo facility's census, as the build renders it: the tier-3 channel
#: database's 2908 device channels, the four tune and chromaticity channels
#: and the user beam mode the top place carries itself, the transfer line's 40 channels, the
#: storage ring injection septa's 7 channels, the injector's 379 channels
#: beyond the frozen ones, the storage ring's beam lifetime and operation
#: mode, and the storage ring's 1390 peripheral channels. A channel that names
#: no signal is neither written nor read: the four tune and chromaticity
#: channels and the user beam mode.
EXPECTED_DEVICES = 1016
EXPECTED_BINDINGS = 4731
EXPECTED_WRITES = 674
EXPECTED_READS = 4052

#: Devices a described group names, each carrying its family's description:
#: every device is in one.
EXPECTED_FAMILY_DESCRIBED = 1016

#: Prose predicates carried once by every ``narad_sem:ChannelBinding``: the
#: channel's own sentence.
BINDING_DESCRIPTION_PREDICATES = ("description",)

#: Prose predicates carried once by every device node, from its system, plus
#: the SYSTEM token itself — the one token the device IRI does not spell, so it
#: ships as data a query can filter on.
DEVICE_DESCRIPTION_PREDICATES = ("systemDescription",)
DEVICE_TOKEN_PREDICATES = ("system",)

#: Prose predicates carried once by every device a described group names.
FAMILY_DESCRIPTION_PREDICATES = ("familyDescription",)

#: n10s' uppercase spellings of the relationship types the shipped example
#: queries use. Any of these in the corpus means the queries miss.
UPPERCASE_N10S_NAMES = ("HASBINDING", "READSSIGNAL", "WRITESSIGNAL", "PARTOF", "LOCATEDIN")


@pytest.fixture(scope="module")
def view_path(built_demo: BuiltProject) -> Path:
    """The graph view the demo build writes."""
    return built_demo.build_dir / "data" / "graph" / "facility.ttl"


@pytest.fixture(scope="module")
def graph(view_path: Path) -> object:
    """The graph view, parsed once."""
    from rdflib import Graph

    parsed = Graph()
    parsed.parse(view_path, format="turtle")
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
    predicate appearing 4731 times spread over 40 bindings would satisfy a
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

    for predicate in FAMILY_DESCRIPTION_PREDICATES:
        assert triples(predicate) == EXPECTED_FAMILY_DESCRIBED, (
            f"narad_p:{predicate} appears {triples(predicate)} times, "
            f"not {EXPECTED_FAMILY_DESCRIBED}."
        )
        assert len(subjects(predicate)) == EXPECTED_FAMILY_DESCRIBED


def test_semantic_signals_carry_no_description(graph) -> None:
    """No prose predicate lands on a ``narad_sem:SemanticSignal``.

    A signal is shared by every channel that reads or writes it, so text on a
    signal could only be one channel's wording standing in for all of them. The
    view puts that text on bindings and devices.
    """
    from rdflib import RDF, URIRef

    signals = set(graph.subjects(RDF.type, URIRef(NARAD_SEMANTICS + "SemanticSignal")))
    assert signals, "The corpus declares no semantic signals at all"

    for predicate in (
        BINDING_DESCRIPTION_PREDICATES
        + DEVICE_DESCRIPTION_PREDICATES
        + FAMILY_DESCRIPTION_PREDICATES
    ):
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


def test_no_uppercase_n10s_names(view_path: Path) -> None:
    text = view_path.read_text(encoding="utf-8")
    for name in UPPERCASE_N10S_NAMES:
        assert name not in text
