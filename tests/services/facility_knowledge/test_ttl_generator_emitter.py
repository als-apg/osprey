"""Tests for the TTL generator's Turtle emitter.

Three things are checked here, in rising order of cost:

1. **Vocabulary spelling** — the emitted text uses the Turtle-side NARAD names
   (``narad_p:hasBinding``, ``narad_p:readsSignal``, …) and never the uppercase
   names neosemantics invents on the Cypher side.
2. **Semantics** — the text parses back to exactly the graph :func:`build_graph`
   builds (``rdflib.compare.isomorphic``), and serialising twice yields the same
   bytes.  Nothing is compared byte-for-byte against a golden file: pre-commit
   normalises files, so a golden would be fragile for no gain.
3. **Predicate isomorphism on real data** — a corpus built from the shipped
   tier-3 channel database carries every predicate the shipped example queries
   touch, with the counts the channel database and the limits file imply.

``rdflib`` is imported at module scope on purpose: it is a core dependency, so
a missing rdflib is a broken test environment rather than a reason to skip and
report a vacuous green.  The *source* modules still import it lazily —
``test_import_isolation.py`` guards that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.compare import isomorphic

from osprey.mcp_server.graph.tools.examples_data import EXAMPLE_QUERIES
from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase
from osprey.services.facility_knowledge.ttl_generator import direction, emitter, model
from osprey.services.facility_knowledge.ttl_generator.ontology_map import (
    OntologyMap,
    load_demo_ontology,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTROL_ASSISTANT_DATA = _REPO_ROOT / "src/osprey/templates/apps/control_assistant/data"
_TIER3_HIERARCHICAL = _CONTROL_ASSISTANT_DATA / "channel_databases/tiers/tier3/hierarchical.json"
_CHANNEL_LIMITS = _CONTROL_ASSISTANT_DATA / "channel_limits.json"

# Facts the model task established about the shipped tier-3 database; the
# direction task established the 396/2512 split (writable iff the address ends
# in ``:SP``).
_REAL_BINDINGS = 2908
_REAL_DEVICES = 512
_REAL_WRITES = 396
_REAL_READS = _REAL_BINDINGS - _REAL_WRITES

_NARAD_P = model.NARAD_PROPERTY_NS
_NARAD_SEM = model.NARAD_SEM_NS

#: The uppercase spellings that belong to the Cypher projection only.
_N10S_ONLY_NAMES = (
    "HASBINDING",
    "READSSIGNAL",
    "WRITESSIGNAL",
    "SUBCLASSOF",
    "ORDINALINSECTION",
    "SOURCENAME",
    "SECTIONCODE",
    "FULLPV",
)

_SYNTHETIC_ADDRESSES = (
    "SR:MAG:DIPOLE:01:CURRENT:SP",
    "SR:MAG:DIPOLE:01:CURRENT:RB",
    "SR:MAG:DIPOLE:02:CURRENT:SP",
    "SR:MAG:HCM:01:CURRENT:SP",
    "SR:DIAG:BPM:01:POSITION:X",
    "SR:DIAG:BPM:01:STATUS:VALID",
    "SR:VAC:GAUGE:SR01:PRESSURE:RB",
    "BR:MAG:QF:01:CURRENT:RB",
)


def _prop(name: str) -> URIRef:
    """URIRef of a ``narad_p:`` property."""
    return URIRef(f"{_NARAD_P}{name}")


def _sem(name: str) -> URIRef:
    """URIRef of a ``narad_sem:`` term."""
    return URIRef(f"{_NARAD_SEM}{name}")


def _directed(graph_model: model.GraphModel) -> model.GraphModel:
    """Stamp directions by the demo-machine rule: a ``:SP`` subfield is a write."""
    return graph_model.with_directions(
        {
            group.key: (
                direction.DIRECTION_WRITE
                if group.subfield == direction.WRITE_SUBFIELD
                else direction.DIRECTION_READ
            )
            for group in graph_model.signal_groups
        }
    )


@pytest.fixture(scope="module")
def ontology() -> OntologyMap:
    """The shipped demo ontology table."""
    return load_demo_ontology()


@pytest.fixture(scope="module")
def synthetic_model() -> model.GraphModel:
    """A small model covering four families, both directions and the gauge naming."""
    return _directed(model.build_model({address: {} for address in _SYNTHETIC_ADDRESSES}))


@pytest.fixture(scope="module")
def synthetic_ttl(synthetic_model: model.GraphModel, ontology: OntologyMap) -> str:
    """Turtle text for the synthetic model."""
    return emitter.serialize_turtle(synthetic_model, ontology)


@pytest.fixture(scope="module")
def real_model() -> model.GraphModel:
    """The full demo machine, with directions resolved from the shipped limits file."""
    database = HierarchicalChannelDatabase(str(_TIER3_HIERARCHICAL))
    built = model.build_model(database.channel_map)
    directed, report = direction.assign_directions(built, _CHANNEL_LIMITS)
    assert report.source is direction.DirectionSource.LIMITS
    return directed


@pytest.fixture(scope="module")
def real_graph(real_model: model.GraphModel, ontology: OntologyMap) -> Graph:
    """The full demo corpus, serialised and parsed back."""
    graph = Graph()
    graph.parse(data=emitter.serialize_turtle(real_model, ontology), format="turtle")
    return graph


# ---------------------------------------------------------------------------
# Vocabulary spelling
# ---------------------------------------------------------------------------


class TestVocabularySpelling:
    """The emitted text must speak Turtle NARAD, never the Cypher projection."""

    def test_declares_exactly_the_narad_prefixes(self, synthetic_ttl: str) -> None:
        """The six prefixes the NARAD convention binds, spelled out rather than
        read off another corpus, so the pin is this test's own."""
        emitted = {line for line in synthetic_ttl.splitlines() if line.startswith("@prefix")}
        assert emitted == {
            "@prefix narad_p: <https://narad.example.org/property/> .",
            "@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        }

    @pytest.mark.parametrize("name", _N10S_ONLY_NAMES)
    def test_no_n10s_uppercase_names_anywhere(self, synthetic_ttl: str, name: str) -> None:
        assert name not in synthetic_ttl, (
            f"{name} is the neosemantics-mapped spelling; the Turtle side uses the "
            "narad_p local names"
        )

    @pytest.mark.parametrize(
        "predicate",
        [
            "narad_p:bindingId",
            "narad_p:confidence",
            "narad_p:deviceId",
            "narad_p:facility",
            "narad_p:fullPv",
            "narad_p:hasBinding",
            "narad_p:ordinalInFacility",
            "narad_p:ordinalInSection",
            "narad_p:protocol",
            "narad_p:rawType",
            "narad_p:readsSignal",
            "narad_p:sPositionM",
            "narad_p:sectionCode",
            "narad_p:sourceName",
            "narad_p:sourceSectionId",
            "narad_p:system",
            "narad_p:writesSignal",
            "rdfs:label",
            "rdfs:subClassOf",
            "skos:altLabel",
        ],
    )
    def test_expected_predicate_is_emitted(self, synthetic_ttl: str, predicate: str) -> None:
        assert f" {predicate} " in synthetic_ttl

    def test_class_and_property_declarations_match_the_shipped_shapes(
        self, synthetic_ttl: str
    ) -> None:
        assert "narad_sem:SemanticSignal a owl:Class ." in synthetic_ttl
        assert "narad_sem:ChannelBinding a owl:Class ;\n    rdfs:subClassOf owl:Thing ." in (
            synthetic_ttl
        )
        assert "narad_sem:AcceleratorDevice a owl:Class ." in synthetic_ttl
        assert "narad_p:readsSignal a owl:ObjectProperty ." in synthetic_ttl
        assert "narad_p:writesSignal a owl:ObjectProperty ." in synthetic_ttl
        # NARAD declares hasBinding as both, so the generated corpus does too.
        assert "narad_p:hasBinding a owl:DatatypeProperty,\n        owl:ObjectProperty ." in (
            synthetic_ttl
        )

    def test_device_iris_are_written_in_full(self, synthetic_ttl: str) -> None:
        # No `narad:` prefix is declared (the NARAD convention binds none), so
        # device and binding IRIs must appear as absolute IRIs.
        assert f"<{model.DEVICE_IRI_PREFIX}demo_SR_DIPOLE01>" in synthetic_ttl
        assert "narad:" not in synthetic_ttl.split("\n\n")[0].replace("narad_p:", "").replace(
            "narad_sem:", ""
        )


# ---------------------------------------------------------------------------
# Node shapes
# ---------------------------------------------------------------------------


class TestNodeShapes:
    """Devices, bindings and signals carry exactly the properties NARAD gives them."""

    @pytest.fixture(scope="class")
    def graph(self, synthetic_model: model.GraphModel, ontology: OntologyMap) -> Graph:
        return emitter.build_graph(synthetic_model, ontology)

    def test_device_carries_the_identity_and_position_properties(
        self, graph: Graph, synthetic_model: model.GraphModel
    ) -> None:
        device = URIRef(model.device_iri("SR", "DIPOLE01"))
        assert (device, URIRef(f"{emitter.RDF_NS}type"), _sem("Dipole")) in graph
        assert (device, _prop("deviceId"), Literal("narad:device:demo:SR:DIPOLE01")) in graph
        assert (device, _prop("sectionCode"), Literal("SR")) in graph
        assert (device, _prop("sourceName"), Literal("DIPOLE01")) in graph
        assert (device, _prop("rawType"), Literal("DIPOLE")) in graph
        # The literal comes off the model, not off a module constant.
        assert (device, _prop("facility"), Literal(synthetic_model.facility)) in graph
        assert len(list(graph.objects(device, _prop("sPositionM")))) == 1
        assert len(list(graph.objects(device, _prop("ordinalInSection")))) == 1
        assert len(list(graph.objects(device, _prop("ordinalInFacility")))) == 1
        assert len(list(graph.objects(device, _prop("hasBinding")))) == 2

    def test_binding_carries_exactly_five_properties_and_one_direction(
        self, graph: Graph, synthetic_model: model.GraphModel
    ) -> None:
        rdf_type = URIRef(f"{emitter.RDF_NS}type")
        for binding in synthetic_model.bindings:
            subject = URIRef(binding.iri)
            predicates = sorted(str(p) for p in graph.predicates(subject, None))
            reads = str(_prop("readsSignal"))
            writes = str(_prop("writesSignal"))
            assert (subject, rdf_type, _sem("ChannelBinding")) in graph
            assert (reads in predicates) != (writes in predicates), (
                f"{binding.full_pv} must carry exactly one of readsSignal/writesSignal"
            )
            expected = {
                str(rdf_type),
                str(_prop("bindingId")),
                str(_prop("confidence")),
                str(_prop("fullPv")),
                str(_prop("protocol")),
                writes if writes in predicates else reads,
            }
            assert set(predicates) == expected

    def test_device_properties_never_land_on_a_binding(
        self, graph: Graph, synthetic_model: model.GraphModel
    ) -> None:
        device_only = ("sourceName", "sectionCode", "sPositionM", "ordinalInSection", "system")
        for binding in synthetic_model.bindings:
            subject = URIRef(binding.iri)
            for name in device_only:
                assert (subject, _prop(name), None) not in graph

    def test_setpoint_writes_and_readback_reads(self, graph: Graph) -> None:
        setpoint = URIRef(model.binding_iri("SR", "DIPOLE01", "CURRENT", "SP"))
        readback = URIRef(model.binding_iri("SR", "DIPOLE01", "CURRENT", "RB"))
        assert (setpoint, _prop("writesSignal"), _sem("dipole_current_sp")) in graph
        assert (readback, _prop("readsSignal"), _sem("dipole_current_rb")) in graph

    def test_signal_individuals_are_typed_and_labelled(
        self, graph: Graph, synthetic_model: model.GraphModel
    ) -> None:
        rdf_type = URIRef(f"{emitter.RDF_NS}type")
        for group in synthetic_model.signal_groups:
            subject = URIRef(group.iri)
            assert (subject, rdf_type, _sem("SemanticSignal")) in graph
            assert (subject, URIRef(emitter.RDFS_LABEL), Literal(group.name)) in graph

    def test_class_hierarchy_is_closed_under_subclassof(
        self, graph: Graph, synthetic_model: model.GraphModel, ontology: OntologyMap
    ) -> None:
        subclass_of = URIRef(emitter.RDFS_SUBCLASS_OF)
        owl_class = URIRef(emitter.OWL_CLASS)
        rdf_type = URIRef(f"{emitter.RDF_NS}type")
        for device in synthetic_model.devices:
            current = ontology.class_for(device.family).name
            hops = 0
            while current != "AcceleratorDevice":
                node = _sem(current)
                assert (node, rdf_type, owl_class) in graph
                parents = list(graph.objects(node, subclass_of))
                assert len(parents) == 1, f"{current} should have exactly one parent"
                current = str(parents[0]).rsplit("/", 1)[-1]
                hops += 1
                assert hops < 10, "subClassOf chain does not terminate"
            assert (_sem("AcceleratorDevice"), rdf_type, owl_class) in graph


# ---------------------------------------------------------------------------
# Determinism and round-trip
# ---------------------------------------------------------------------------


class TestDeterminismAndRoundTrip:
    """Same input, same bytes; and the bytes mean what the graph means."""

    def test_serialising_twice_is_byte_identical(
        self, synthetic_model: model.GraphModel, ontology: OntologyMap
    ) -> None:
        first = emitter.serialize_turtle(synthetic_model, ontology)
        second = emitter.serialize_turtle(synthetic_model, ontology)
        assert first == second
        assert first.splitlines() == second.splitlines()

    def test_rebuilding_the_model_reproduces_the_same_text(self, ontology: OntologyMap) -> None:
        # A different dict insertion order must not change the output.
        forward = _directed(model.build_model({a: {} for a in _SYNTHETIC_ADDRESSES}))
        backward = _directed(model.build_model({a: {} for a in reversed(_SYNTHETIC_ADDRESSES)}))
        assert emitter.serialize_turtle(forward, ontology) == emitter.serialize_turtle(
            backward, ontology
        )

    def test_text_parses_back_to_the_built_graph(
        self, synthetic_model: model.GraphModel, ontology: OntologyMap, synthetic_ttl: str
    ) -> None:
        parsed = Graph()
        parsed.parse(data=synthetic_ttl, format="turtle")
        built = emitter.build_graph(synthetic_model, ontology)
        assert len(parsed) == len(built)
        assert isomorphic(parsed, built), "serialised text and built graph disagree"

    def test_output_ends_in_a_single_newline(self, synthetic_ttl: str) -> None:
        assert synthetic_ttl.endswith("\n")
        assert not synthetic_ttl.endswith("\n\n")

    def test_no_blank_nodes(self, synthetic_model: model.GraphModel, ontology: OntologyMap) -> None:
        graph = emitter.build_graph(synthetic_model, ontology)
        assert not any(
            term.startswith("_:") for triple in graph for term in (str(triple[0]), str(triple[2]))
        )


# ---------------------------------------------------------------------------
# Errors and file writing
# ---------------------------------------------------------------------------


class TestErrorsAndWriting:
    """An undirected model is refused; a written file round-trips."""

    def test_undirected_group_is_refused_with_an_actionable_message(
        self, ontology: OntologyMap
    ) -> None:
        undirected = model.build_model({"SR:MAG:DIPOLE:01:CURRENT:SP": {}})
        with pytest.raises(emitter.UndirectedSignalError) as excinfo:
            emitter.serialize_turtle(undirected, ontology)
        message = str(excinfo.value)
        assert "DIPOLE:CURRENT:SP" in message
        assert "assign_directions" in message

    def test_unknown_direction_is_refused(self, ontology: OntologyMap) -> None:
        built = model.build_model({"SR:MAG:DIPOLE:01:CURRENT:SP": {}})
        bogus = built.with_directions({group.key: "sideways" for group in built.signal_groups})
        with pytest.raises(ValueError, match="sideways"):
            emitter.serialize_turtle(bogus, ontology)

    def test_unknown_family_is_refused_by_the_ontology_table(self, ontology: OntologyMap) -> None:
        built = _directed(model.build_model({"SR:MAG:UNOBTAINIUM:01:CURRENT:SP": {}}))
        with pytest.raises(KeyError, match="UNOBTAINIUM"):
            emitter.serialize_turtle(built, ontology)

    def test_write_turtle_creates_parents_and_returns_the_path(
        self, synthetic_model: model.GraphModel, ontology: OntologyMap, tmp_path: Path
    ) -> None:
        target = tmp_path / "nested" / "demo.ttl"
        returned = emitter.write_turtle(synthetic_model, ontology, target)
        assert returned == target
        assert target.read_text(encoding="utf-8") == emitter.serialize_turtle(
            synthetic_model, ontology
        )
        parsed = Graph()
        parsed.parse(source=str(target), format="turtle")
        assert isomorphic(parsed, emitter.build_graph(synthetic_model, ontology))

    def test_write_turtle_writes_utf8(
        self, synthetic_model: model.GraphModel, ontology: OntologyMap, tmp_path: Path
    ) -> None:
        target = emitter.write_turtle(synthetic_model, ontology, tmp_path / "demo.ttl")
        assert target.read_bytes().decode("utf-8").startswith("@prefix narad_p:")


# ---------------------------------------------------------------------------
# Real data: the shipped tier-3 channel database
# ---------------------------------------------------------------------------


class TestShippedDemoMachine:
    """The generated corpus must be able to answer every shipped example query."""

    def test_binding_and_device_counts(self, real_graph: Graph) -> None:
        assert len(list(real_graph.subject_objects(_prop("hasBinding")))) == _REAL_BINDINGS
        assert len(list(real_graph.subject_objects(_prop("writesSignal")))) == _REAL_WRITES
        assert len(list(real_graph.subject_objects(_prop("readsSignal")))) == _REAL_READS
        assert len(set(real_graph.subjects(_prop("deviceId"), None))) == _REAL_DEVICES
        assert len(set(real_graph.subjects(_prop("fullPv"), None))) == _REAL_BINDINGS

    def test_every_writable_binding_is_a_setpoint(self, real_graph: Graph) -> None:
        written = {
            str(next(real_graph.objects(subject, _prop("fullPv"))))
            for subject in real_graph.subjects(_prop("writesSignal"), None)
        }
        assert len(written) == _REAL_WRITES
        assert all(pv.endswith(":SP") for pv in written)

    def test_limits_file_and_corpus_agree_on_the_writable_set(self, real_graph: Graph) -> None:
        writable = direction.load_writable_set(_CHANNEL_LIMITS)
        written = {
            str(next(real_graph.objects(subject, _prop("fullPv"))))
            for subject in real_graph.subjects(_prop("writesSignal"), None)
        }
        assert written == set(writable)

    def test_every_device_property_the_examples_read_is_present(self, real_graph: Graph) -> None:
        for name in ("sourceName", "sectionCode", "sPositionM", "ordinalInSection"):
            assert len(set(real_graph.subjects(_prop(name), None))) == _REAL_DEVICES, name

    def test_every_binding_property_the_examples_read_is_present(self, real_graph: Graph) -> None:
        for name in ("fullPv", "confidence", "protocol", "bindingId"):
            assert len(set(real_graph.subjects(_prop(name), None))) == _REAL_BINDINGS, name

    def test_class_hierarchy_is_traversable(self, real_graph: Graph) -> None:
        subclass_of = URIRef(emitter.RDFS_SUBCLASS_OF)
        edges = list(real_graph.subject_objects(subclass_of))
        # Every declared device class except the root has exactly one parent, and
        # ChannelBinding hangs off owl:Thing exactly as the NARAD convention has it.
        assert (URIRef(emitter.CHANNEL_BINDING_CLASS_IRI), URIRef(emitter.OWL_THING)) in edges
        magnet = _sem("Magnet")
        assert (magnet, _sem("AcceleratorDevice")) in edges
        assert (_sem("Dipole"), magnet) in edges
        assert (_sem("HCorrector"), _sem("Corrector")) in edges
        assert (_sem("Corrector"), magnet) in edges

    @pytest.mark.parametrize("example", EXAMPLE_QUERIES, ids=lambda ex: ex.key)
    def test_demo_parameters_exist_in_the_generated_corpus(
        self, real_graph: Graph, example: object
    ) -> None:
        """Every ``$parameter`` value the demo corpus advertises must be real.

        An example whose parameter names a device, section, PV or class that the
        generated corpus does not hold would return zero rows — the failure mode
        this test exists to catch.
        """
        params = example.parameters  # type: ignore[attr-defined]
        rdf_type = URIRef(f"{emitter.RDF_NS}type")
        for name, value in params.items():
            if name == "section":
                assert (None, _prop("sectionCode"), Literal(value)) in real_graph
            elif name == "name":
                assert (None, _prop("sourceName"), Literal(value)) in real_graph
            elif name == "pv":
                assert (None, _prop("fullPv"), Literal(value)) in real_graph
            elif name in {"class_uri", "root_uri"}:
                assert (URIRef(value), rdf_type, URIRef(emitter.OWL_CLASS)) in real_graph
            else:  # pragma: no cover - a new parameter kind needs a rule here
                pytest.fail(f"example {example.key} has an unhandled parameter {name!r}")

    def test_q3_device_and_q6_pv_are_connected(self, real_graph: Graph) -> None:
        """q3 names DIPOLE01/SR and q6 names its setpoint — they must be one device."""
        device = URIRef(model.device_iri("SR", "DIPOLE01"))
        binding = URIRef(model.binding_iri("SR", "DIPOLE01", "CURRENT", "SP"))
        assert (device, _prop("hasBinding"), binding) in real_graph
        assert (
            binding,
            _prop("fullPv"),
            Literal("SR:MAG:DIPOLE:01:CURRENT:SP"),
        ) in real_graph
        assert (binding, _prop("writesSignal"), None) in real_graph

    def test_ontology_covers_every_family_in_the_database(self, ontology: OntologyMap) -> None:
        raw = json.loads(_TIER3_HIERARCHICAL.read_text(encoding="utf-8"))
        families = {
            family
            for ring, systems in raw["tree"].items()
            if not ring.startswith("_")
            for system, fams in systems.items()
            if not system.startswith("_")
            for family in fams
            if not family.startswith("_")
        }
        for family in sorted(families):
            assert ontology.class_for(family) is not None


# ---------------------------------------------------------------------------
# Description predicates
# ---------------------------------------------------------------------------

#: Hierarchy prose for the described fixture — one text per level, ring-qualified.
_HIERARCHY_TEXTS = model.HierarchyDescriptions(
    ring={("SR",): "Storage ring"},
    system={("SR", "MAG"): "Storage ring magnet system"},
    family={("SR", "MAG", "DIPOLE"): "Storage ring bending dipoles"},
    field={("SR", "MAG", "DIPOLE", "CURRENT"): "Dipole coil current"},
    subfield={("SR", "MAG", "DIPOLE", "CURRENT", "SP"): "Commanded dipole current"},
)

#: Per-channel prose, keyed by the six-token address exactly as in_context.json is.
_BINDING_TEXTS = {
    "SR:MAG:DIPOLE:01:CURRENT:SP": "Storage ring dipole 01 current setpoint",
    "SR:MAG:DIPOLE:01:CURRENT:RB": "Storage ring dipole 01 current readback",
}

#: Two SR channels the fixture describes and one BR channel it deliberately does
#: not — the undescribed one proves an exact join leaves neighbours alone.
_DESCRIBED_ADDRESSES = (
    "SR:MAG:DIPOLE:01:CURRENT:SP",
    "SR:MAG:DIPOLE:01:CURRENT:RB",
    "BR:MAG:QF:01:CURRENT:RB",
)

#: The six prose predicates, in the order the emitter declares them.
_DESCRIPTION_NAMES = (
    emitter.P_DESCRIPTION,
    emitter.P_FIELD_DESCRIPTION,
    emitter.P_SUBFIELD_DESCRIPTION,
    emitter.P_FAMILY_DESCRIPTION,
    emitter.P_SYSTEM_DESCRIPTION,
    emitter.P_RING_DESCRIPTION,
)


def _build_described() -> model.GraphModel:
    """The described fixture model, built fresh so insertion order can be varied."""
    return _directed(
        model.build_model(
            {address: {} for address in _DESCRIBED_ADDRESSES},
            hierarchy_descriptions=_HIERARCHY_TEXTS,
            binding_descriptions=_BINDING_TEXTS,
        )
    )


def _block(ttl: str, rendered_subject: str) -> list[str]:
    """Return the lines of one subject block, found by its rendered subject term."""
    for block in ttl.split("\n\n"):
        if block.startswith(f"{rendered_subject} "):
            return block.splitlines()
    raise AssertionError(f"{rendered_subject} has no block in the emitted Turtle")


class TestDescriptionPredicates:
    """Prose reaches bindings and devices, under six distinct one-value predicates."""

    @pytest.fixture(scope="class")
    def described_model(self) -> model.GraphModel:
        return _build_described()

    @pytest.fixture(scope="class")
    def described_ttl(self, described_model: model.GraphModel, ontology: OntologyMap) -> str:
        return emitter.serialize_turtle(described_model, ontology)

    @pytest.fixture(scope="class")
    def described_graph(self, described_model: model.GraphModel, ontology: OntologyMap) -> Graph:
        return emitter.build_graph(described_model, ontology)

    def test_described_binding_renders_the_exact_literal_lines(self, described_ttl: str) -> None:
        setpoint = model.binding_iri("SR", "DIPOLE01", "CURRENT", "SP")
        lines = _block(described_ttl, f"<{setpoint}>")
        assert '    narad_p:description "Storage ring dipole 01 current setpoint" ;' in lines
        assert '    narad_p:fieldDescription "Dipole coil current" ;' in lines
        assert '    narad_p:subfieldDescription "Commanded dipole current" ;' in lines

    def test_described_device_renders_the_exact_literal_lines(self, described_ttl: str) -> None:
        lines = _block(described_ttl, f"<{model.device_iri('SR', 'DIPOLE01')}>")
        assert '    narad_p:familyDescription "Storage ring bending dipoles" ;' in lines
        assert '    narad_p:ringDescription "Storage ring" ;' in lines
        assert '    narad_p:systemDescription "Storage ring magnet system" .' in lines

    def test_each_prose_predicate_carries_exactly_one_value_per_node(
        self, described_graph: Graph, described_model: model.GraphModel
    ) -> None:
        # n10s collapses repeated values of one predicate, so two texts under one
        # name would silently lose data on import.
        for name in _DESCRIPTION_NAMES:
            for subject in set(described_graph.subjects(_prop(name), None)):
                assert len(list(described_graph.objects(subject, _prop(name)))) == 1, name
        binding = URIRef(model.binding_iri("SR", "DIPOLE01", "CURRENT", "SP"))
        device = URIRef(model.device_iri("SR", "DIPOLE01"))
        assert set(described_graph.predicates(binding, None)) == {
            URIRef(f"{emitter.RDF_NS}type"),
            _prop("bindingId"),
            _prop("confidence"),
            _prop("description"),
            _prop("fieldDescription"),
            _prop("fullPv"),
            _prop("protocol"),
            _prop("subfieldDescription"),
            _prop("writesSignal"),
        }
        assert {
            _prop("familyDescription"),
            _prop("systemDescription"),
            _prop("ringDescription"),
        } <= set(described_graph.predicates(device, None))
        assert described_model.devices  # the fixture really did build devices

    def test_an_exact_join_leaves_the_undescribed_nodes_bare(self, described_graph: Graph) -> None:
        binding = URIRef(model.binding_iri("BR", "QF01", "CURRENT", "RB"))
        device = URIRef(model.device_iri("BR", "QF01"))
        for name in _DESCRIPTION_NAMES:
            assert (binding, _prop(name), None) not in described_graph, name
            assert (device, _prop(name), None) not in described_graph, name

    def test_semantic_signals_carry_no_prose(
        self, described_graph: Graph, described_model: model.GraphModel
    ) -> None:
        allowed = {URIRef(f"{emitter.RDF_NS}type"), URIRef(emitter.RDFS_LABEL)}
        for group in described_model.signal_groups:
            subject = URIRef(group.iri)
            assert set(described_graph.predicates(subject, None)) == allowed

    def test_a_model_without_prose_emits_no_description_clause(
        self, synthetic_model: model.GraphModel, synthetic_ttl: str, ontology: OntologyMap
    ) -> None:
        graph = emitter.build_graph(synthetic_model, ontology)
        for name in _DESCRIPTION_NAMES:
            assert (None, _prop(name), None) not in graph, name
        # Every prose clause is indented; only the vocabulary block may name the
        # predicates, and there they are subjects at column zero.
        assert [
            line
            for line in synthetic_ttl.splitlines()
            if line.startswith("    narad_p:") and "escription" in line
        ] == []

    def test_prose_does_not_disturb_determinism(self, ontology: OntologyMap) -> None:
        forward = _build_described()
        first = emitter.serialize_turtle(forward, ontology)
        assert emitter.serialize_turtle(forward, ontology) == first
        backward = _directed(
            model.build_model(
                {address: {} for address in reversed(_DESCRIBED_ADDRESSES)},
                hierarchy_descriptions=_HIERARCHY_TEXTS,
                binding_descriptions=dict(reversed(list(_BINDING_TEXTS.items()))),
            )
        )
        assert emitter.serialize_turtle(backward, ontology) == first

    def test_described_text_parses_back_to_the_built_graph(
        self, described_model: model.GraphModel, ontology: OntologyMap, described_ttl: str
    ) -> None:
        parsed = Graph()
        parsed.parse(data=described_ttl, format="turtle")
        assert isomorphic(parsed, emitter.build_graph(described_model, ontology))

    @pytest.mark.parametrize("name", _DESCRIPTION_NAMES)
    def test_every_prose_predicate_is_declared_in_the_vocabulary(
        self, synthetic_ttl: str, name: str
    ) -> None:
        assert name in emitter.PROPERTY_NAMES
        assert name not in emitter.OBJECT_PROPERTY_NAMES
        assert f"narad_p:{name} a owl:DatatypeProperty ." in synthetic_ttl


# ---------------------------------------------------------------------------
# The system token
# ---------------------------------------------------------------------------


class TestSystemProperty:
    """Every device carries its system token under ``narad_p:system``."""

    @pytest.fixture(scope="class")
    def graph(self, synthetic_model: model.GraphModel, ontology: OntologyMap) -> Graph:
        return emitter.build_graph(synthetic_model, ontology)

    def test_every_device_carries_exactly_one_system_literal(
        self, graph: Graph, synthetic_model: model.GraphModel
    ) -> None:
        assert synthetic_model.devices  # the fixture really did build devices
        for device in synthetic_model.devices:
            objects = list(graph.objects(URIRef(device.iri), _prop("system")))
            assert objects == [Literal(device.system)], device.address

    def test_system_renders_the_exact_literal_line(
        self, synthetic_ttl: str, ontology: OntologyMap
    ) -> None:
        # Sorted by IRI, ``system`` is a device's last clause unless the device
        # also carries prose, where ``systemDescription`` follows it.
        magnet = _block(synthetic_ttl, f"<{model.device_iri('SR', 'DIPOLE01')}>")
        assert '    narad_p:system "MAG" .' in magnet
        vacuum = _block(synthetic_ttl, f"<{model.device_iri('SR', 'GAUGESR01')}>")
        assert '    narad_p:system "VAC" .' in vacuum
        described = emitter.serialize_turtle(_build_described(), ontology)
        lines = _block(described, f"<{model.device_iri('SR', 'DIPOLE01')}>")
        assert '    narad_p:system "MAG" ;' in lines
        assert '    narad_p:systemDescription "Storage ring magnet system" .' in lines

    def test_system_is_declared_as_a_datatype_property(self, synthetic_ttl: str) -> None:
        assert "system" in emitter.PROPERTY_NAMES
        assert "system" not in emitter.OBJECT_PROPERTY_NAMES
        assert "narad_p:system a owl:DatatypeProperty ." in synthetic_ttl

    def test_the_system_token_is_not_the_system_description(self, graph: Graph) -> None:
        # Two predicates, two meanings: the token a query filters on, and the
        # ring-qualified prose.  A model without prose carries only the token.
        device = URIRef(model.device_iri("SR", "DIPOLE01"))
        assert (device, _prop("system"), Literal("MAG")) in graph
        assert (device, _prop("systemDescription"), None) not in graph

    def test_system_emission_stays_deterministic(self, ontology: OntologyMap) -> None:
        forward = _directed(model.build_model({address: {} for address in _SYNTHETIC_ADDRESSES}))
        first = emitter.serialize_turtle(forward, ontology)
        assert emitter.serialize_turtle(forward, ontology) == first
        backward = _directed(
            model.build_model({address: {} for address in reversed(_SYNTHETIC_ADDRESSES)})
        )
        assert emitter.serialize_turtle(backward, ontology) == first

    def test_system_text_parses_back_to_the_built_graph(
        self, synthetic_model: model.GraphModel, ontology: OntologyMap, synthetic_ttl: str
    ) -> None:
        parsed = Graph()
        parsed.parse(data=synthetic_ttl, format="turtle")
        assert isomorphic(parsed, emitter.build_graph(synthetic_model, ontology))

    def test_every_device_in_the_shipped_corpus_carries_a_system(
        self, real_graph: Graph, real_model: model.GraphModel
    ) -> None:
        subjects = set(real_graph.subjects(_prop("system"), None))
        assert len(subjects) == _REAL_DEVICES
        assert subjects == {URIRef(device.iri) for device in real_model.devices}


# ---------------------------------------------------------------------------
# The facility token
# ---------------------------------------------------------------------------


def _facility_model(token: str) -> model.GraphModel:
    """The synthetic model, minted for *token* rather than for the demo machine."""
    return _directed(
        model.build_model({address: {} for address in _SYNTHETIC_ADDRESSES}, facility=token)
    )


class TestFacilityLiteral:
    """``narad_p:facility`` is read off the model, so it cannot disagree with the IRIs.

    Before the token became an input, the emitter held its own copy of the
    module constant.  Patching only ``model.FACILITY`` then produced a corpus
    whose IRIs said one facility and whose ``narad_p:facility`` literal said
    another — no error, just a graph query that joins nothing.
    """

    @pytest.fixture(scope="class")
    def custom_model(self) -> model.GraphModel:
        return _facility_model("xyz")

    @pytest.fixture(scope="class")
    def custom_ttl(self, custom_model: model.GraphModel, ontology: OntologyMap) -> str:
        return emitter.serialize_turtle(custom_model, ontology)

    def test_the_emitter_holds_no_facility_of_its_own(self) -> None:
        # The second place the value could come from is exactly the bug.
        assert not hasattr(emitter, "FACILITY")

    def test_every_device_carries_the_models_token(
        self, custom_model: model.GraphModel, ontology: OntologyMap
    ) -> None:
        graph = emitter.build_graph(custom_model, ontology)
        assert custom_model.devices
        for device in custom_model.devices:
            objects = list(graph.objects(URIRef(device.iri), _prop("facility")))
            assert objects == [Literal("xyz")], device.address

    def test_the_demo_token_appears_nowhere_in_a_custom_corpus(self, custom_ttl: str) -> None:
        assert '    narad_p:facility "xyz" ;' in custom_ttl
        assert "demo" not in custom_ttl

    def test_the_default_corpus_is_unchanged(self, synthetic_ttl: str) -> None:
        assert '    narad_p:facility "demo" ;' in synthetic_ttl

    def test_two_facilities_in_one_process_each_serialise_consistently(
        self, ontology: OntologyMap
    ) -> None:
        alpha = emitter.serialize_turtle(_facility_model("alpha"), ontology)
        beta = emitter.serialize_turtle(_facility_model("beta"), ontology)
        assert "beta" not in alpha
        assert "alpha" not in beta
        # Same corpus, one token apart — no shared state leaked between builds.
        assert alpha.replace("alpha", "TOKEN") == beta.replace("beta", "TOKEN")

    def test_custom_text_parses_back_to_the_built_graph(
        self, custom_model: model.GraphModel, ontology: OntologyMap, custom_ttl: str
    ) -> None:
        parsed = Graph()
        parsed.parse(data=custom_ttl, format="turtle")
        assert isomorphic(parsed, emitter.build_graph(custom_model, ontology))


# ---------------------------------------------------------------------------
# Facility-supplied extra properties
# ---------------------------------------------------------------------------

#: One device and one binding of the synthetic fixture, by address.
_EXTRA_DEVICE = "SR:MAG:DIPOLE:01"
_EXTRA_BINDING = "SR:MAG:DIPOLE:01:CURRENT:SP"

#: One value of each type the carrier accepts, deliberately not in sorted order.
_DEVICE_EXTRAS = {"serialNumber": "SN-4711", "rackHeightU": 4, "crateSlot": 1.5}
_BINDING_EXTRAS = {"units": "A", "displayPrecision": 3}


def _extras_model(**kwargs) -> model.GraphModel:
    """The synthetic model, carrying the facility's own scalar properties."""
    return _directed(model.build_model({address: {} for address in _SYNTHETIC_ADDRESSES}, **kwargs))


def _with_extras() -> model.GraphModel:
    return _extras_model(
        device_properties={_EXTRA_DEVICE: _DEVICE_EXTRAS},
        binding_properties={_EXTRA_BINDING: _BINDING_EXTRAS},
    )


class TestExtraProperties:
    """A facility's own scalar attributes become first-class ``narad_p:`` properties."""

    @pytest.fixture(scope="class")
    def extras_model(self) -> model.GraphModel:
        return _with_extras()

    @pytest.fixture(scope="class")
    def extras_ttl(self, extras_model: model.GraphModel, ontology: OntologyMap) -> str:
        return emitter.serialize_turtle(extras_model, ontology)

    @pytest.fixture(scope="class")
    def extras_graph(self, extras_model: model.GraphModel, ontology: OntologyMap) -> Graph:
        return emitter.build_graph(extras_model, ontology)

    def test_device_extras_render_with_the_literal_form_of_their_type(
        self, extras_ttl: str
    ) -> None:
        lines = _block(extras_ttl, f"<{model.device_iri('SR', 'DIPOLE01')}>")
        assert "    narad_p:crateSlot 1.5 ;" in lines
        assert "    narad_p:rackHeightU 4 ;" in lines
        assert '    narad_p:serialNumber "SN-4711" ;' in lines

    def test_binding_extras_render_with_the_literal_form_of_their_type(
        self, extras_ttl: str
    ) -> None:
        lines = _block(extras_ttl, f"<{model.binding_iri('SR', 'DIPOLE01', 'CURRENT', 'SP')}>")
        assert "    narad_p:displayPrecision 3 ;" in lines
        assert '    narad_p:units "A" ;' in lines

    def test_extras_appear_in_sorted_key_order(self, extras_ttl: str) -> None:
        lines = _block(extras_ttl, f"<{model.device_iri('SR', 'DIPOLE01')}>")
        positions = [
            next(
                index for index, line in enumerate(lines) if line.startswith(f"    narad_p:{key} ")
            )
            for key in sorted(_DEVICE_EXTRAS)
        ]
        assert positions == sorted(positions)

    def test_the_graph_carries_one_typed_object_per_extra(self, extras_graph: Graph) -> None:
        device = URIRef(model.device_iri("SR", "DIPOLE01"))
        binding = URIRef(model.binding_iri("SR", "DIPOLE01", "CURRENT", "SP"))
        assert list(extras_graph.objects(device, _prop("serialNumber"))) == [Literal("SN-4711")]
        assert list(extras_graph.objects(device, _prop("rackHeightU"))) == [Literal(4)]
        assert list(extras_graph.objects(binding, _prop("displayPrecision"))) == [Literal(3)]
        assert list(extras_graph.objects(binding, _prop("units"))) == [Literal("A")]

    def test_only_the_named_nodes_carry_them(
        self, extras_graph: Graph, extras_model: model.GraphModel
    ) -> None:
        assert set(extras_graph.subjects(_prop("serialNumber"), None)) == {
            URIRef(model.device_iri("SR", "DIPOLE01"))
        }
        assert set(extras_graph.subjects(_prop("units"), None)) == {
            URIRef(model.binding_iri("SR", "DIPOLE01", "CURRENT", "SP"))
        }
        assert extras_model.devices  # the fixture really did build devices

    def test_extras_are_declared_as_datatype_properties(self, extras_ttl: str) -> None:
        for name in (*_DEVICE_EXTRAS, *_BINDING_EXTRAS):
            assert f"narad_p:{name} a owl:DatatypeProperty ." in extras_ttl

    def test_the_built_in_constant_is_the_floor_and_stays_untouched(
        self, extras_model: model.GraphModel
    ) -> None:
        declared = emitter.declared_property_names(extras_model)
        assert declared[: len(emitter.PROPERTY_NAMES)] == emitter.PROPERTY_NAMES
        assert declared[len(emitter.PROPERTY_NAMES) :] == tuple(
            sorted((*_DEVICE_EXTRAS, *_BINDING_EXTRAS))
        )
        for name in (*_DEVICE_EXTRAS, *_BINDING_EXTRAS):
            assert name not in emitter.PROPERTY_NAMES

    def test_a_model_without_extras_declares_exactly_the_built_ins(
        self, synthetic_model: model.GraphModel
    ) -> None:
        assert emitter.declared_property_names(synthetic_model) == emitter.PROPERTY_NAMES

    def test_an_empty_carrier_adds_no_bytes(self, ontology: OntologyMap) -> None:
        """The seed marker is a checksum over these bytes, so empty must cost nothing."""
        baseline = emitter.serialize_turtle(_extras_model(), ontology)
        empty = emitter.serialize_turtle(
            _extras_model(device_properties={}, binding_properties={}), ontology
        )
        unmatched = emitter.serialize_turtle(
            _extras_model(device_properties={"BR:MAG:SEXTUPOLE:99": {"crate": "nowhere"}}),
            ontology,
        )
        assert empty == baseline
        assert unmatched == baseline

    def test_extras_do_not_disturb_determinism(self, ontology: OntologyMap) -> None:
        first = emitter.serialize_turtle(_with_extras(), ontology)
        assert emitter.serialize_turtle(_with_extras(), ontology) == first
        backward = _directed(
            model.build_model(
                {address: {} for address in reversed(_SYNTHETIC_ADDRESSES)},
                device_properties={_EXTRA_DEVICE: dict(reversed(list(_DEVICE_EXTRAS.items())))},
                binding_properties={_EXTRA_BINDING: dict(reversed(list(_BINDING_EXTRAS.items())))},
            )
        )
        assert emitter.serialize_turtle(backward, ontology) == first

    def test_extras_text_parses_back_to_the_built_graph(
        self, extras_model: model.GraphModel, ontology: OntologyMap, extras_ttl: str
    ) -> None:
        parsed = Graph()
        parsed.parse(data=extras_ttl, format="turtle")
        built = emitter.build_graph(extras_model, ontology)
        assert len(parsed) == len(built)
        assert isomorphic(parsed, built), "serialised text and built graph disagree"

    @pytest.mark.parametrize("name", ["fullPv", "system"])
    def test_a_key_colliding_with_a_built_in_property_is_refused(self, name: str) -> None:
        """A duplicate predicate with a different meaning is worse than a failure."""
        with pytest.raises(model.ExtraPropertyError, match=name):
            _extras_model(binding_properties={_EXTRA_BINDING: {name: "x"}})
