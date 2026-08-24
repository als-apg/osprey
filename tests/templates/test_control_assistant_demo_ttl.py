"""The control-assistant preset seeds its graph store from the demo machine.

Two files have to agree for a fresh `osprey up` to bring up a graph the agent
can actually search: the `services.graphdb.ttl_path` in the preset's
`config.yml.j2`, and the corpus that path names. Nothing else checks that pair
— the deploy reads the corpus at seed time and only warns when it is missing,
so a preset pointing at a file the render does not ship would deploy an empty
store and say nothing about it. That is the first test here: render the preset
the way the CLI does, resolve the key the way the deploy does, and require the
file to be on disk.

The rest pins what is *in* that corpus. It is generated — `osprey knowledge
build-ttl` derives it from the channel database in the same `data/` tree — and
a generated file that nobody counts can drift silently: a change to the channel
database, the direction pass, or the emitter would quietly ship a different
graph. The census below (512 devices, 2908 bindings, 396 written and 2512 read
signals) is the demo machine as of the corpus committed beside this test, and
the 396 writes are exactly its `:SP` addresses. The prose census sits beside it:
three description predicates on every binding, three more plus the SYSTEM token
on every device, and none of the six on a semantic signal.

The uppercase check guards the other half of the pipeline. neosemantics imports
a predicate IRI under the local name it finds, so `narad_p:hasBinding` becomes
the relationship type `hasBinding` — but n10s also has a `LABELS_AND_NODES`
handling that would uppercase it to `HASBINDING`, and the example queries the
agent is given all spell the camelCase form. A corpus carrying uppercase names
would import into a graph where every shipped query returns nothing.

Finally, ariel_standalone is asserted to still name `als_gtb.ttl`: the two
presets deliberately seed different corpora, and the repoint of one must not
quietly become a repoint of both.
"""

from pathlib import Path

import pytest
import yaml

from osprey.cli.templates.manager import TemplateManager

#: Where the demo corpus lands in a rendered project, relative to the rendered
#: ``config.yml``. Spelled here rather than read from the config so the test
#: states the expected value instead of agreeing with whatever is configured.
EXPECTED_TTL_PATH = "./data/demo_machine.ttl"

#: The corpus as it ships in the preset's data tree.
TEMPLATE_TTL = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/demo_machine.ttl"
)

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


def _render_project(name: str, bundle: str, tmp_path: Path) -> Path:
    """Render a project from an app preset, the way ``osprey build`` does.

    ``TemplateManager.create_project`` is the renderer the CLI calls; going
    through it rather than through ``osprey build`` skips the venv and the
    lifecycle without skipping any of the templating or the data copy, which is
    what this file is about.
    """
    return TemplateManager().create_project(
        project_name=name,
        output_dir=tmp_path,
        data_bundle=bundle,
        context={"channel_finder_mode": "hierarchical"},
    )


def _graphdb_block(project_dir: Path) -> dict:
    """The rendered ``services.graphdb`` block, through the deploy's resolver."""
    from osprey.deployment.graphdb_service import resolve_graphdb_service_config

    config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
    # Raises on a malformed block, so this also asserts the repoint left the
    # block in a shape the deploy preflight accepts.
    resolve_graphdb_service_config(config)
    return config["services"]["graphdb"]


@pytest.fixture(scope="module")
def control_assistant_project(tmp_path_factory) -> Path:
    return _render_project(
        "demo-ttl-ca", "control_assistant", tmp_path_factory.mktemp("demo_ttl_ca")
    )


@pytest.fixture(scope="module")
def graph() -> object:
    """The committed corpus, parsed once."""
    from rdflib import Graph

    parsed = Graph()
    parsed.parse(TEMPLATE_TTL, format="turtle")
    return parsed


# ---------------------------------------------------------------------------
# The preset points at a corpus the render actually ships
# ---------------------------------------------------------------------------


def test_rendered_ttl_path_is_the_demo_corpus(control_assistant_project: Path) -> None:
    assert _graphdb_block(control_assistant_project)["ttl_path"] == EXPECTED_TTL_PATH


def test_configured_corpus_is_on_disk_in_a_rendered_project(
    control_assistant_project: Path,
) -> None:
    """Resolve ``ttl_path`` exactly as the deploy's seeding step does.

    ``_graphdb_config_dir`` is the deploy's rule for *what* a relative value is
    relative to (the render one zone down when there is one, the project root
    otherwise); ``resolve_bundle_path`` is the shared rule for resolving it.
    """
    from osprey.deployment.container_lifecycle import _graphdb_config_dir
    from osprey.services.facility_knowledge.bundle_path import resolve_bundle_path

    ttl_path = _graphdb_block(control_assistant_project)["ttl_path"]
    resolved = resolve_bundle_path(ttl_path, _graphdb_config_dir(control_assistant_project))

    assert resolved.is_file(), (
        f"services.graphdb.ttl_path names {resolved}, which the rendered project "
        "does not ship — the graph store would deploy empty."
    )
    assert resolved.read_bytes() == TEMPLATE_TTL.read_bytes()


def test_ariel_standalone_still_seeds_the_als_corpus(tmp_path: Path) -> None:
    project = _render_project("demo-ttl-ariel", "ariel_standalone", tmp_path)
    assert _graphdb_block(project)["ttl_path"].endswith("als_gtb.ttl")


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
