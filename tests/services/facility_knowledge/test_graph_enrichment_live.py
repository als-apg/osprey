"""Live proof that the enriched corpora answer the questions they were enriched for.

Every other test of the corpus enrichment reads Turtle: it can prove a triple was
written, which is not the same as proving an operator can ask for it.  Two steps
sit between the two claims and neither is visible from the TTL:

* **n10s decides the property shape.**  ``skos:altLabel`` reaches Cypher as a
  list only because :data:`~...graph_seeder.N10S_GRAPH_CONFIG` asks for
  ``handleMultival='ARRAY'`` restricted to that one predicate.  Under the plain
  ``OVERWRITE`` setting the same TTL lands with one synonym per subject and the
  rest silently dropped, so ``ANY(l IN s.altLabel WHERE l = 'beam current')`` —
  the shape every synonym lookup needs — has nothing to iterate.  A file-level
  test cannot tell those two stores apart.
* **A store that is already configured keeps its old config.**  n10s refuses to
  re-initialize one, so changing the canonical dict does not reconfigure the
  graph an operator already has: :func:`~...graph_seeder.bootstrap` reports
  ``DIFFERS`` and the operator has to come back through
  ``osprey knowledge seed-graph --force``.  That is a real migration with a real
  failure mode, and it is only observable against a live n10s.

So this lane seeds the shipped corpus through the seeder API and asks the
questions in Cypher.  It is separate from ``tests/integration/test_graphdb_store.py``
— which pins the corpus's verified node counts — because it owns a different
claim (the *enrichment* is reachable) and because its tests wipe the store
between steps, which would strand that module's session-scoped fixture.

Each test starts from a wiped store and seeds the corpus itself — which is also
the sequence ``seed-graph --force`` performs, so the wipe is under test rather
than around it.

The container recipe (pinned image, n10s jar from the pinned release or
``OSPREY_TEST_N10S_JAR``, APOC copied out of the image) comes from
:mod:`tests._graphdb_container` rather than being copied; see that module's
docstring for why the container starts without ``NEO4J_PLUGINS``.  The store is
module-scoped because this module wipes it; the jars it mounts are resolved once
per session by the shared ``graphdb_plugin_dir`` fixture, which is also the skip
gate — without Docker, without the image, or without the jar this module skips
and says which.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from tests._graphdb_container import (
    GRAPHDB_TEST_PASSWORD,
    GRAPHDB_TEST_USERNAME,
    graphdb_store,
)

logger = logging.getLogger(__name__)

# xdist_group("docker") keeps every container-starting module on one xdist
# worker: two testcontainers sessions starting at once race their own reapers
# over the port publisher.  ``integration`` because this needs a real service.
pytestmark = [pytest.mark.integration, pytest.mark.xdist_group("docker")]


# ---------------------------------------------------------------------------
# Subjects the assertions name
# ---------------------------------------------------------------------------

#: A demo binding whose three description predicates are all populated, chosen
#: because it is a setpoint: the subfield description is the one that says
#: "read-write", so a wrong-subfield regression reads as wrong prose here.
DEMO_BINDING_PV = "SR:MAG:DIPOLE:01:CURRENT:SP"

#: A demo device, addressed the way the shipped Cypher examples address one.
DEMO_DEVICE_NAME = "BPM01"
DEMO_DEVICE_SECTION = "SR"

#: The SYSTEM token that device carries.  Diagnostics rather than magnets on
#: purpose: SYSTEM is the one address token that is *not* recoverable from the
#: device's class, so a device whose system and family disagree is the case that
#: proves the token was emitted rather than inferred.
DEMO_DEVICE_SYSTEM = "DIAG"

_SEMANTICS = "https://narad.example.org/schema/shared_semantics/"

#: A demo ontology class carrying several synonyms.
DEMO_MULTI_LABEL_CLASS = f"{_SEMANTICS}BeamPositionMonitor"
DEMO_MULTI_LABEL_SYNONYM = "bpm"

#: A demo ontology class carrying exactly **one** synonym.  Load-bearing: a
#: single-valued property is where ARRAY and OVERWRITE produce the same *content*
#: and different *shapes*, so this is the subject that proves the list is a
#: property of the config rather than of how many labels happened to be present.
DEMO_SINGLE_LABEL_CLASS = f"{_SEMANTICS}Vacuum"
DEMO_SINGLE_LABEL_SYNONYM = "vacuum"

#: The n10s configuration a store bootstrapped before the array change carries.
#: Spelled out rather than derived from the canonical dict: the point of the
#: migration test is that these two differ, and a copy-with-edits would keep
#: agreeing with the canonical dict as it moved.
LEGACY_GRAPH_CONFIG: dict[str, Any] = {
    "handleVocabUris": "MAP",
    "handleMultival": "OVERWRITE",
    "keepLangTag": False,
    "handleRDFTypes": "LABELS_AND_NODES",
    "applyNeo4jNaming": True,
}


# ---------------------------------------------------------------------------
# Cypher
# ---------------------------------------------------------------------------

#: Do *all* bindings carry the three description predicates, or only the one the
#: point assertion names?  ``count(expr)`` skips nulls, so four numbers that agree
#: is the whole "every description predicate is queryable" claim in one row.
BINDING_DESCRIPTION_COVERAGE = """
MATCH (b:ChannelBinding)
RETURN count(b)                       AS total,
       count(b.description)           AS described,
       count(b.fieldDescription)      AS field_described,
       count(b.subfieldDescription)   AS subfield_described
"""

BINDING_DESCRIPTIONS = """
MATCH (b:ChannelBinding {fullPv: $pv})
RETURN b.description         AS description,
       b.fieldDescription    AS field_description,
       b.subfieldDescription AS subfield_description
"""

#: A device is a resource carrying at least one binding — the same definition the
#: store's own count query uses.
DEVICE_DESCRIPTION_COVERAGE = """
MATCH (d:Resource)-[:HASBINDING]->(:ChannelBinding)
WITH DISTINCT d
RETURN count(d)                     AS total,
       count(d.familyDescription)   AS family_described,
       count(d.systemDescription)   AS system_described,
       count(d.ringDescription)     AS ring_described,
       count(d.system)              AS with_system
"""

DEVICE_DETAIL = """
MATCH (d:Resource {sourceName: $name, sectionCode: $section})
RETURN d.system              AS system,
       d.familyDescription   AS family_description,
       d.systemDescription   AS system_description,
       d.ringDescription     AS ring_description
"""

#: The question the prose was added to answer: find channels by what they do
#: rather than by how they are spelled.
BINDING_DESCRIPTION_SEARCH = """
MATCH (b:ChannelBinding)
WHERE toLower(b.description) CONTAINS $needle
RETURN count(b) AS n
"""

#: No description predicate may reach a semantic signal.  A signal is a shared
#: concept across every device that carries it, so device- or channel-specific
#: prose on one would be a lie about all the others.
SIGNAL_DESCRIPTION_LEAK = """
MATCH (s:SemanticSignal)
WHERE s.description IS NOT NULL
   OR s.fieldDescription IS NOT NULL
   OR s.subfieldDescription IS NOT NULL
   OR s.familyDescription IS NOT NULL
   OR s.systemDescription IS NOT NULL
   OR s.ringDescription IS NOT NULL
RETURN count(s) AS n
"""

SIGNAL_COUNT = "MATCH (s:SemanticSignal) RETURN count(s) AS n"

ALT_LABELS_FOR_URI = "MATCH (r:Resource {uri: $uri}) RETURN r.altLabel AS alt_labels"

#: The list-semantics query the synonym lookups are built on.  It is not merely a
#: nicer spelling of an equality test: run against a scalar ``altLabel`` this
#: raises rather than returning nothing, which is why the shape assertion above
#: it has to be explicit.
CLASS_BY_SYNONYM = """
MATCH (c:Class)
WHERE ANY(label IN c.altLabel WHERE label = $synonym)
RETURN c.uri AS uri
"""

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def enrichment_store_uri(graphdb_plugin_dir: Path) -> Iterator[str]:
    """Start a throwaway graph store for this module and yield its bolt URI.

    Module-scoped rather than session-scoped: the tests below wipe the store
    between steps, and a store shared with another module's session-scoped
    fixture would hand that module an emptied graph halfway through its run.
    The plugins it mounts *are* session-scoped — see ``graphdb_plugin_dir``,
    which is also this module's skip gate.
    """
    with graphdb_store(graphdb_plugin_dir) as uri:
        yield uri


@pytest.fixture(scope="module")
def demo_ttl() -> str:
    """The generated demo corpus, read the way installed code reads it."""
    resource = (
        files("osprey.templates")
        .joinpath("apps")
        .joinpath("control_assistant")
        .joinpath("data")
        .joinpath("demo_machine.ttl")
    )
    return resource.read_text(encoding="utf-8")


@pytest.fixture
def clean_store(enrichment_store_uri: str) -> Iterator[Any]:
    """Yield a driver session onto an emptied, unconfigured store.

    ``wipe`` takes the n10s graph config and namespace prefixes with the data, so
    every test that receives this fixture starts from the state a fresh deploy
    starts from and has to bootstrap for itself.
    """
    from osprey.services.facility_knowledge.seeder import graph_seeder

    with graph_seeder.open_session(
        enrichment_store_uri, GRAPHDB_TEST_USERNAME, GRAPHDB_TEST_PASSWORD
    ) as session:
        graph_seeder.wipe(session)
        yield session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(session: Any, cypher: str, **params: Any) -> Any:
    """Run a query expected to return exactly one row and return it."""
    record = session.run(cypher, **params).single()
    assert record is not None, f"query returned no row: {cypher}"
    return record


def _ttl_census(ttl: str) -> dict[str, int]:
    """Count the subjects a corpus declares, straight out of its Turtle text.

    Cheap string counts rather than an rdflib parse, and derived from the corpus
    under test rather than pinned as numbers here: the committed censuses belong
    to ``tests/templates/test_control_assistant_demo_ttl.py``, and a second copy
    of them would go stale on its own schedule.  What this lane needs is only the
    comparison — whether the graph holds *everything* the file declares, which is
    what turns "the descriptions are consistent" into "the corpus is complete".
    Without it a batch that died a tenth of the way in would satisfy every
    coverage assertion below, since a truncated import is internally consistent.

    ``narad_p:deviceId "`` carries the trailing quote so it counts the devices
    that *use* the property and not the one line declaring it in the vocabulary.
    """
    return {
        "bindings": ttl.count("a narad_sem:ChannelBinding"),
        "devices": ttl.count('narad_p:deviceId "'),
        "signals": ttl.count("a narad_sem:SemanticSignal"),
    }


def _seed(session: Any, ttl: str, label: str) -> None:
    """Bootstrap a wiped store and import *ttl* into it, as ``seed-graph`` does."""
    from osprey.services.facility_knowledge.seeder import graph_seeder

    result = graph_seeder.bootstrap(session)
    assert result.status is graph_seeder.BootstrapStatus.INITIALIZED, result.message

    imported = graph_seeder.import_ttl(session, ttl)
    assert imported.termination_status == graph_seeder.TERMINATION_OK, (
        f"n10s refused the {label} corpus: {imported.extra_info}"
    )
    assert imported.triples_loaded == imported.triples_parsed, (
        f"n10s parsed {imported.triples_parsed} triples out of the {label} corpus "
        f"but committed {imported.triples_loaded}"
    )
    graph_seeder.write_marker(
        session,
        graph_seeder.ttl_sha256(ttl),
        graph_seeder.parse_direction_source(ttl),
    )


def _assert_alt_labels(value: Any, *, subject: str, expected: str) -> None:
    """Assert *value* is a Cypher list of synonyms containing *expected*.

    ``isinstance(value, list)`` is the assertion, not a precondition for a nicer
    error message: a scalar carrying the right synonym is precisely the shape
    ``handleMultival='OVERWRITE'`` produces, and every equality test that could
    be written about it would pass.
    """
    assert isinstance(value, list), (
        f"{subject} came back with altLabel={value!r} ({type(value).__name__}). "
        "Synonym lookups iterate this property, so a scalar means the store was "
        "configured without handleMultival='ARRAY' and every synonym but one was "
        "dropped on import"
    )
    assert expected in value, f"{subject} carries {value!r}, which does not include {expected!r}"


# ---------------------------------------------------------------------------
# The demo corpus
# ---------------------------------------------------------------------------


def test_the_demo_corpus_answers_description_and_system_questions(
    clean_store: Any, demo_ttl: str
) -> None:
    """Seed the generated demo corpus and ask it everything the enrichment added.

    One test over one seeded store rather than five over five, deliberately: the
    corpus takes tens of seconds to import and every assertion below is a
    question about the *same* graph, so splitting them would re-seed the same
    2.7 MB per question and prove nothing extra.
    """
    session = clean_store
    _seed(session, demo_ttl, "demo")
    census = _ttl_census(demo_ttl)

    # --- Every binding carries all three description predicates -------------
    bindings = _row(session, BINDING_DESCRIPTION_COVERAGE)
    total = bindings["total"]
    assert total == census["bindings"], (
        f"the store holds {total} channel bindings and the corpus declares "
        f"{census['bindings']} — the import did not finish, so every coverage "
        "number below would agree about a fraction of the machine"
    )
    assert bindings["described"] == total, (
        f"{total - bindings['described']} of {total} bindings have no description"
    )
    assert bindings["field_described"] == total
    assert bindings["subfield_described"] == total

    # --- and a named one carries the prose an operator would recognise ------
    detail = _row(session, BINDING_DESCRIPTIONS, pv=DEMO_BINDING_PV)
    for key in ("description", "field_description", "subfield_description"):
        value = detail[key]
        assert isinstance(value, str) and value.strip(), (
            f"{DEMO_BINDING_PV} came back with {key}={value!r}"
        )
    assert "dipole" in detail["description"].lower()
    assert "read-write" in detail["subfield_description"].lower(), (
        "the subfield description of a setpoint is what tells a reader the channel is writable"
    )

    # --- Every device carries its three descriptions and its SYSTEM token ---
    devices = _row(session, DEVICE_DESCRIPTION_COVERAGE)
    device_total = devices["total"]
    assert device_total == census["devices"], (
        f"the store holds {device_total} devices and the corpus declares {census['devices']}"
    )
    assert devices["family_described"] == device_total
    assert devices["system_described"] == device_total
    assert devices["ring_described"] == device_total
    assert devices["with_system"] == device_total, (
        f"{device_total - devices['with_system']} of {device_total} devices carry "
        "no system token, so a question scoped to one system cannot be answered"
    )

    device = _row(session, DEVICE_DETAIL, name=DEMO_DEVICE_NAME, section=DEMO_DEVICE_SECTION)
    assert device["system"] == DEMO_DEVICE_SYSTEM
    for key in ("family_description", "system_description", "ring_description"):
        value = device[key]
        assert isinstance(value, str) and value.strip(), (
            f"{DEMO_DEVICE_SECTION}/{DEMO_DEVICE_NAME} came back with {key}={value!r}"
        )

    # --- The prose is searchable, which is the point of putting it there ----
    found = _row(session, BINDING_DESCRIPTION_SEARCH, needle="beam position")["n"]
    assert found > 0, (
        "no binding description mentions 'beam position' — the descriptions "
        "imported as opaque values rather than as searchable text"
    )

    # --- Class synonyms are lists, at both cardinalities --------------------
    multi = _row(session, ALT_LABELS_FOR_URI, uri=DEMO_MULTI_LABEL_CLASS)["alt_labels"]
    _assert_alt_labels(multi, subject=DEMO_MULTI_LABEL_CLASS, expected=DEMO_MULTI_LABEL_SYNONYM)
    assert len(multi) > 1, (
        f"{DEMO_MULTI_LABEL_CLASS} kept only {multi!r}; the corpus gives it several synonyms"
    )

    single = _row(session, ALT_LABELS_FOR_URI, uri=DEMO_SINGLE_LABEL_CLASS)["alt_labels"]
    _assert_alt_labels(single, subject=DEMO_SINGLE_LABEL_CLASS, expected=DEMO_SINGLE_LABEL_SYNONYM)

    by_synonym = [
        record["uri"] for record in session.run(CLASS_BY_SYNONYM, synonym=DEMO_MULTI_LABEL_SYNONYM)
    ]
    assert DEMO_MULTI_LABEL_CLASS in by_synonym

    # --- and no description predicate reached a semantic signal -------------
    assert _row(session, SIGNAL_COUNT)["n"] == census["signals"], (
        "the semantic signals are the subjects the leak query below scans; a "
        "short count there would make its zero meaningless"
    )
    leaked = _row(session, SIGNAL_DESCRIPTION_LEAK)["n"]
    assert leaked == 0, (
        f"{leaked} semantic signals carry a description predicate. A signal is "
        "shared by every device that reads or writes it, so prose about one "
        "channel on a signal is wrong for all the others"
    )


# ---------------------------------------------------------------------------
# The migration
# ---------------------------------------------------------------------------


def _legacy_bootstrap(session: Any) -> None:
    """Configure a wiped store the way one bootstrapped before the array change.

    Deliberately not :func:`~...graph_seeder.bootstrap`: the whole point is a
    store whose config is *not* osprey's current one, which is the state every
    graph store deployed before this change is sitting in.
    """
    from osprey.services.facility_knowledge.seeder import graph_seeder

    session.run(graph_seeder._CONSTRAINT_CYPHER).consume()
    session.run("CALL n10s.graphconfig.init($params)", params=LEGACY_GRAPH_CONFIG).consume()
    for prefix, namespace in graph_seeder.NARAD_PREFIXES.items():
        session.run(
            "CALL n10s.nsprefixes.add($prefix, $namespace)", prefix=prefix, namespace=namespace
        ).consume()


def test_a_legacy_store_is_reported_then_recovered_by_force(
    clean_store: Any, demo_ttl: str
) -> None:
    """Walk a pre-change store through the whole migration and back.

    One test rather than four because each step's precondition is the previous
    step's outcome: ``DIFFERS`` can only be observed on a store that was
    configured the old way, and ``ALREADY_CANONICAL`` only means something on a
    store this test just re-initialized.  The multi-synonym class is the subject
    because its synonyms are the data the old configuration actually damaged, so
    the first and last assertions read the same subject on either side of the
    migration.
    """
    from osprey.services.facility_knowledge.seeder import graph_seeder

    session = clean_store

    # --- 1. A store configured the old way collapses the synonyms ----------
    _legacy_bootstrap(session)
    imported = graph_seeder.import_ttl(session, demo_ttl)
    assert imported.termination_status == graph_seeder.TERMINATION_OK, imported.extra_info

    collapsed = _row(session, ALT_LABELS_FOR_URI, uri=DEMO_MULTI_LABEL_CLASS)["alt_labels"]
    assert isinstance(collapsed, str), (
        f"the legacy configuration was expected to leave one scalar synonym on "
        f"{DEMO_MULTI_LABEL_CLASS}, but it came back as {collapsed!r}. If n10s no longer "
        "collapses multi-valued predicates under handleMultival='OVERWRITE', "
        "this migration has nothing to recover and the test is obsolete"
    )

    # --- 2. bootstrap() reports the drift instead of re-initializing -------
    # n10s hard-refuses to re-initialize a configured store, so this is the only
    # thing bootstrap *can* do: an operator has to come back through --force.
    drifted = graph_seeder.bootstrap(session)
    assert drifted.status is graph_seeder.BootstrapStatus.DIFFERS, drifted.message
    assert not drifted.ok
    assert "handleMultival" in drifted.differing_keys, (
        f"the reported drift was {drifted.differing_keys}, which does not name the key that changed"
    )
    assert "--force" in drifted.message

    # --- 3. The --force path: wipe, re-init, re-import ---------------------
    graph_seeder.wipe(session)
    assert graph_seeder.resource_count(session) == 0
    assert graph_seeder.read_marker(session) is None

    _seed(session, demo_ttl, "demo")

    recovered = _row(session, ALT_LABELS_FOR_URI, uri=DEMO_MULTI_LABEL_CLASS)["alt_labels"]
    _assert_alt_labels(recovered, subject=DEMO_MULTI_LABEL_CLASS, expected=DEMO_MULTI_LABEL_SYNONYM)
    assert len(recovered) > 1

    # --- 4. and the store is now canonical, permanently --------------------
    settled = graph_seeder.bootstrap(session)
    assert settled.status is graph_seeder.BootstrapStatus.ALREADY_CANONICAL, settled.message
    assert settled.differing_keys == ()
    assert settled.ok
    assert graph_seeder.read_marker(session) == graph_seeder.ttl_sha256(demo_ttl), (
        "re-bootstrapping a canonical store must not disturb the seed marker"
    )
