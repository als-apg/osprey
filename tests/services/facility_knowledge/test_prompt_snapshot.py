"""Unit tests for the seed-time prompt snapshot.

Three claims, each with the failure it guards against:

1. **Collection is the shared one.** ``collect_schema`` filters bookkeeping and
   skips unquotable labels exactly as the ``get_schema`` tool documents —
   asserted on the *queries issued*, not just the payload, because a collection
   that read the seed marker and filtered afterwards would leak on a store
   whose bookkeeping moves.
2. **The block renders each example runnable.** Every example arrives with its
   Cypher and the parameter set that runs it, or an explicit ``none``.
3. **Applying is idempotent and refuses unmarked files.** The bake runs on
   every ``osprey up``, so a block that grew on each application — or one
   appended into a hand-edited prompt without markers — would corrupt renders
   at the deploy cadence.

The template ships the markers this module rewrites, so their spellings are
pinned against each other here: a drift between the two turns every bake into
a silent no-op, which is exactly the placeholder-forever failure the seeder
owns preventing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.services.facility_knowledge.seeder import prompt_snapshot as mod
from osprey.utils.workspace import BUILD_DIR_NAME, IMAGE_DIR_NAME

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# A recording Cypher seam
# ---------------------------------------------------------------------------

_DEFAULT_LABELS = ("Class", "Resource", "_GraphConfig", "_NsPrefDef", "_OspreySeed")
_DEFAULT_RELTYPES = ("HASBINDING", "READSSIGNAL", "SUBCLASSOF", "TYPE")
_DEFAULT_KEYS: dict[str, list[str]] = {
    "Class": ["label", "uri"],
    "Resource": ["fullPv", "sourceName", "uri"],
}

#: Two classes, deliberately unalike: one whose IRI separates on ``/`` and
#: answers with the ARRAY the store's n10s config produces (with a duplicate, so
#: deduplication is exercised), one separating on ``#`` whose ``altLabel`` comes
#: back as a bare scalar — the shape a hand-configured store can answer with.
_DEFAULT_VOCABULARY: tuple[dict[str, Any], ...] = (
    {
        "uri": "https://narad.example.org/schema/shared_semantics/VacuumGauge",
        "synonyms": ["vgc", "gauge", "vgc"],
    },
    {
        "uri": "https://narad.example.org/schema/shared_semantics#Thermocouple",
        "synonyms": "tc",
    },
)

#: One device with a binding, as the specimen query returns it.
_DEFAULT_SPECIMEN: dict[str, Any] = {
    "name": "SR01U-VGC1",
    "section": "SR01",
    "system": "vacuum",
    "pv": "SR01U:VGC1:PRESSURE",
}


@dataclass
class _FakeStore:
    """Routes on the Cypher text and records every query, like the tool's fake."""

    labels: tuple[str, ...] = _DEFAULT_LABELS
    relationship_types: tuple[str, ...] = _DEFAULT_RELTYPES
    keys_by_label: dict[str, list[str]] = field(default_factory=lambda: dict(_DEFAULT_KEYS))
    vocabulary: tuple[dict[str, Any], ...] = _DEFAULT_VOCABULARY
    specimen: dict[str, Any] | None = field(default_factory=lambda: dict(_DEFAULT_SPECIMEN))
    queries: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)

    def run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((cypher, params))
        if "db.labels" in cypher:
            return [{"label": name} for name in self.labels]
        if "db.relationshipTypes" in cypher:
            return [{"relationshipType": name} for name in self.relationship_types]
        if "UNWIND keys(n)" in cypher:
            label = cypher.split("`")[1]
            return [{"key": key} for key in self.keys_by_label.get(label, [])]
        # Matched whole, not by substring: routing on a fragment would keep
        # answering a query the module had since rewritten, and the capture the
        # test believes it is exercising would drift from the one shipped.
        if cypher == mod.VOCABULARY_CYPHER:
            return [dict(row) for row in self.vocabulary]
        if cypher == mod.SPECIMEN_VALUES_CYPHER:
            return [dict(self.specimen)] if self.specimen else []
        raise AssertionError(f"unexpected query: {cypher!r}")


def _raises(cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """A store that is up enough to be dialled and answers nothing but errors."""
    raise RuntimeError("store said no")


@dataclass
class _SchemaOnlyStore(_FakeStore):
    """Answers the schema and errors on both new captures.

    The realistic partial failure: a store seeded by an older ``osprey``, or one
    whose n10s config never landed ``altLabel``, can fail these two queries
    outright rather than answer them empty.
    """

    def run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if cypher in (mod.VOCABULARY_CYPHER, mod.SPECIMEN_VALUES_CYPHER):
            self.queries.append((cypher, params))
            raise RuntimeError("store said no")
        return super().run(cypher, params)


# ---------------------------------------------------------------------------
# collect_schema
# ---------------------------------------------------------------------------


class TestCollectSchema:
    def test_full_capture_issues_unbounded_scans(self):
        """sample_size=None walks without a LIMIT — the seed-time shape."""
        store = _FakeStore()
        schema = mod.collect_schema(store.run)

        scans = [(c, p) for c, p in store.queries if "UNWIND keys(n)" in c]
        assert scans, "no property scans were issued at all"
        for cypher, params in scans:
            assert "LIMIT" not in cypher
            assert params is None
        assert schema["naming"]["sample_size"] is None

    def test_bounded_capture_matches_the_tool_contract(self):
        """A sample_size bounds every scan with the $k the tool documents."""
        store = _FakeStore()
        schema = mod.collect_schema(store.run, sample_size=200)

        scans = [(c, p) for c, p in store.queries if "UNWIND keys(n)" in c]
        for cypher, params in scans:
            assert "WITH n LIMIT $k" in cypher
            assert params == {"k": 200}
        assert schema["naming"]["sample_size"] == 200

    def test_bookkeeping_labels_are_neither_listed_nor_scanned(self):
        store = _FakeStore()
        schema = mod.collect_schema(store.run)

        assert set(schema["labels"]) == {"Class", "Resource"}
        scanned = {c.split("`")[1] for c, _p in store.queries if "UNWIND keys(n)" in c}
        assert scanned.isdisjoint(mod.BOOKKEEPING_LABELS), (
            "the capture read a bookkeeping label's properties"
        )

    def test_bookkeeping_properties_are_filtered_from_knowledge_labels(self):
        store = _FakeStore(keys_by_label={"Resource": ["fullPv", "sha256", "seededAt", "kind"]})
        schema = mod.collect_schema(store.run)
        assert schema["properties_by_label"]["Resource"] == ["fullPv"]

    def test_backtick_label_is_listed_but_not_scanned(self):
        store = _FakeStore(labels=("Resource", "Weird`Label"))
        schema = mod.collect_schema(store.run)

        assert "Weird`Label" in schema["labels"]
        assert "Weird`Label" not in schema["properties_by_label"]


# ---------------------------------------------------------------------------
# collect_vocabulary / resolve_example_values
# ---------------------------------------------------------------------------


class TestCollectVocabulary:
    """The class synonyms are the facility's, read back out of its own ontology."""

    def test_rows_become_named_classes_with_sorted_unique_synonyms(self):
        store = _FakeStore()
        vocabulary = mod.collect_vocabulary(store.run)

        assert [entry["name"] for entry in vocabulary] == ["VacuumGauge", "Thermocouple"], (
            "the display name is the IRI local part, for both / and # separators"
        )
        assert vocabulary[0]["synonyms"] == ["gauge", "vgc"], "not sorted, or not deduplicated"

    def test_a_scalar_altlabel_is_wrapped_not_exploded(self):
        """A single synonym must render as one synonym, not one per character."""
        vocabulary = mod.collect_vocabulary(_FakeStore().run)
        assert vocabulary[1]["synonyms"] == ["tc"]

    def test_a_raising_store_yields_no_vocabulary_and_warns(self, caplog):
        """bake_snapshot runs inside `osprey up`: this query may not break it."""
        with caplog.at_level("WARNING"):
            assert mod.collect_vocabulary(_raises) == []
        assert [r for r in caplog.records if r.levelname == "WARNING"], "no warning was logged"


class TestResolveExampleValues:
    def test_the_specimen_row_becomes_the_values(self):
        values = mod.resolve_example_values(_FakeStore().run)
        assert values == _DEFAULT_SPECIMEN

    def test_a_null_column_is_dropped_rather_than_rendered(self):
        """A device with no `system` must not put a null into an example."""
        specimen = dict(_DEFAULT_SPECIMEN, system=None)
        values = mod.resolve_example_values(_FakeStore(specimen=specimen).run)
        assert "system" not in values
        assert values["name"] == _DEFAULT_SPECIMEN["name"]

    def test_a_store_without_a_bound_device_resolves_nothing(self):
        assert mod.resolve_example_values(_FakeStore(specimen=None).run) == {}

    def test_a_raising_store_yields_no_values_and_warns(self, caplog):
        with caplog.at_level("WARNING"):
            assert mod.resolve_example_values(_raises) == {}
        assert [r for r in caplog.records if r.levelname == "WARNING"], "no warning was logged"


# ---------------------------------------------------------------------------
# render_block / apply_snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Example:
    key: str = "q1a"
    title: str = "Device count by concrete class"
    description: str = "How many devices of each concrete kind the graph holds."
    cypher: str = "MATCH (d:Resource) RETURN count(d) LIMIT 100"
    parameters: dict[str, Any] = field(default_factory=lambda: {"x": 2})


def _schema() -> dict[str, Any]:
    return {
        "labels": ["Class", "Resource"],
        "relationship_types": ["HASBINDING"],
        "properties_by_label": {"Resource": ["fullPv", "uri"]},
        "prefixes": {"narad_p": "https://narad.example.org/property/"},
        "naming": {"note": "a note", "sample_size": None},
    }


def _block(**overrides: Any) -> str:
    kwargs: dict[str, Any] = {
        "digest": "abcdef0123456789",
        "resource_count": 512,
    }
    kwargs.update(overrides)
    examples = kwargs.pop("examples", [_Example()])
    return mod.render_block(_schema(), examples, **kwargs)


class TestRenderBlock:
    def test_block_is_marker_delimited(self):
        block = _block()
        assert block.startswith(mod.SNAPSHOT_BEGIN)
        assert block.endswith(mod.SNAPSHOT_END)

    def test_provenance_names_digest_and_count(self):
        block = _block()
        assert "abcdef012345" in block, "the 12-char digest stamp is missing"
        assert "512 Resource" in block

    def test_unmanaged_store_is_stamped_as_such(self):
        assert "no seed marker" in _block(digest=None)

    def test_parameters_render_with_their_example(self):
        block = _block()
        assert 'Parameters: `{"x": 2}` — framework defaults' in block

    def test_a_parameterless_example_says_none(self):
        example = _Example(parameters={})
        block = mod.render_block(
            _schema(), [example], digest="abcdef0123456789", resource_count=512
        )
        assert "Parameters: none" in block

    def test_schema_and_example_render(self):
        block = _block()
        assert "HASBINDING" in block
        assert "```cypher" in block
        assert "MATCH (d:Resource) RETURN count(d) LIMIT 100" in block


class TestRenderVocabulary:
    """The synonyms the agent will match on are this corpus's, or absent."""

    def test_each_class_gets_a_row_with_its_synonyms(self):
        block = _block(vocabulary=mod.collect_vocabulary(_FakeStore().run))

        assert "### Vocabulary" in block
        assert "| Class | Synonyms (altLabel) |" in block
        assert "| `VacuumGauge` | `gauge`, `vgc` |" in block
        assert "| `Thermocouple` | `tc` |" in block

    def test_the_matching_rule_is_stated_with_the_table(self):
        """`altLabel` is a list; `= $term` against a list is a confident zero."""
        block = _block(vocabulary=mod.collect_vocabulary(_FakeStore().run))
        assert "ANY(l IN c.altLabel WHERE toLower(l) CONTAINS $term)" in block
        assert "never compare with `=`" in block

    def test_an_empty_capture_renders_the_note_not_an_empty_table(self):
        """An empty table would claim the corpus has no synonyms; it may have."""
        block = _block(vocabulary=())
        assert "### Vocabulary" in block
        assert mod.NO_VOCABULARY_NOTE in block
        assert "| Class | Synonyms (altLabel) |" not in block


class TestParameterLabelling:
    """No parameter set is unlabelled: the agent must be able to tell them apart."""

    def test_a_fully_resolved_example_is_labelled_captured(self):
        example = _Example(parameters={"name": "SHIPPED", "section": "SHIPPED"})
        block = _block(examples=[example], values=_DEFAULT_SPECIMEN)

        assert '"name": "SR01U-VGC1"' in block
        assert '"section": "SR01"' in block
        assert f"— {mod.CAPTURED_PARAMETERS_NOTE}" in block
        assert "SHIPPED" not in block

    def test_an_unresolvable_parameter_keeps_its_literal_and_the_default_label(self):
        """`phrase` is deliberately English — resolving it would be meaningless."""
        example = _Example(parameters={"phrase": "vacuum gauge"})
        block = _block(examples=[example], values=_DEFAULT_SPECIMEN)

        assert '"phrase": "vacuum gauge"' in block
        assert f"— {mod.DEFAULT_PARAMETERS_NOTE}" in block

    def test_a_partly_resolved_example_is_labelled_a_default(self):
        """One shipped literal left in the set means the whole set needs swapping."""
        example = _Example(parameters={"name": "SHIPPED", "phrase": "vacuum gauge"})
        block = _block(examples=[example], values=_DEFAULT_SPECIMEN)

        assert '"name": "SR01U-VGC1"' in block
        assert f"— {mod.DEFAULT_PARAMETERS_NOTE}" in block
        assert mod.CAPTURED_PARAMETERS_NOTE not in block

    def test_a_store_that_resolved_nothing_still_renders_every_parameter(self):
        """The fallback is the shipped literal, labelled — never an empty set."""
        from osprey.mcp_server.graph.tools.examples_data import EXAMPLE_QUERIES

        block = _block(examples=EXAMPLE_QUERIES, values={})

        assert "Parameters: `{}`" not in block
        for example in EXAMPLE_QUERIES:
            for value in example.parameters.values():
                assert json.dumps(value) in block
        assert mod.CAPTURED_PARAMETERS_NOTE not in block

    def test_a_parameterless_example_is_not_labelled_at_all(self):
        block = _block(examples=[_Example(parameters={})], values=_DEFAULT_SPECIMEN)
        assert block.count("Parameters: none") == 1
        assert mod.CAPTURED_PARAMETERS_NOTE not in block

    def test_the_intro_names_what_is_captured_and_what_is_not(self):
        block = _block()
        assert "*captured*" in block and "*framework defaults*" in block
        assert "must be swapped for values from this corpus" in block


class TestDirectionProvenance:
    """How this corpus's read/write edges were derived — or nothing at all."""

    def test_limits_derived_edges_say_so(self):
        block = _block(direction_source="limits")
        assert "Direction provenance: read/write edges derived from this facility's" in block
        assert "channel limits" in block

    def test_grammar_derived_edges_say_why_they_are_grammar_derived(self):
        block = _block(direction_source="grammar")
        assert "derived from the address grammar" in block
        assert "no channel limits were available" in block

    def test_an_unrecognised_source_is_printed_verbatim(self):
        """A newer builder's spelling beats silence."""
        assert "Direction provenance: `handmade`." in _block(direction_source="handmade")

    def test_an_older_corpus_recorded_nothing_and_gets_no_line(self):
        """There is no honest default: the two derivations are not interchangeable."""
        assert "Direction provenance" not in _block(direction_source=None)

    @pytest.mark.parametrize("source", [None, "limits", "grammar", "handmade"])
    def test_no_variant_ever_names_a_limits_file(self, source):
        """The prompt-surface guard forbids that bigram anywhere in the render."""
        assert "limits file" not in _block(direction_source=source).lower()


PLACEHOLDER = (
    f"---\nname: facility-knowledge-graph\n---\n\n# Agent\n\n{mod.SNAPSHOT_HEADING}\n\n"
    f"{mod.SNAPSHOT_BEGIN}\n\nNo snapshot yet.\n\n{mod.SNAPSHOT_END}\n\n## Submitting\n"
)


def _render(tmp_path: Path, *personas: str, images: tuple[str, ...] = ()) -> Path:
    """A render with the placeholder agent file at every place the bake looks.

    The render's own agents, one attached persona render per *personas*, and
    one staged image build context per *images*.
    """
    agent_dirs = [tmp_path / ".claude" / "agents"]
    agent_dirs += [tmp_path / persona / ".claude" / "agents" for persona in personas]
    agent_dirs += [
        tmp_path / IMAGE_DIR_NAME / image / BUILD_DIR_NAME / ".claude" / "agents"
        for image in images
    ]
    for agents in agent_dirs:
        agents.mkdir(parents=True)
        (agents / mod.AGENT_FILENAME).write_text(PLACEHOLDER, encoding="utf-8")
    return tmp_path


class _Record:
    def __init__(self, row: dict[str, Any]) -> None:
        self._row = row

    def data(self) -> dict[str, Any]:
        return self._row


def _session(store: _FakeStore) -> Any:
    """The fake store behind the driver-session seam ``bake_snapshot`` reads."""
    return SimpleNamespace(
        run=lambda cypher, params=None: [_Record(row) for row in store.run(cypher, params)]
    )


class TestApplySnapshot:
    def test_replaces_the_managed_region_only(self, tmp_path: Path):
        render = _render(tmp_path)
        patched = mod.apply_snapshot(render, {mod.AGENT_FILENAME: _block()})

        text = (render / ".claude" / "agents" / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        assert patched == [render / ".claude" / "agents" / mod.AGENT_FILENAME]
        assert "No snapshot yet." not in text
        assert "512 Resource" in text
        assert text.startswith("---\nname: facility-knowledge-graph"), "frontmatter was touched"
        assert text.endswith("## Submitting\n"), "text after the region was touched"

    def test_applying_twice_is_applying_once(self, tmp_path: Path):
        render = _render(tmp_path)
        mod.apply_snapshot(render, {mod.AGENT_FILENAME: _block()})
        once = (render / ".claude" / "agents" / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        mod.apply_snapshot(render, {mod.AGENT_FILENAME: _block()})
        twice = (render / ".claude" / "agents" / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        assert once == twice

    def test_a_new_capture_replaces_the_old_one(self, tmp_path: Path):
        render = _render(tmp_path)
        mod.apply_snapshot(render, {mod.AGENT_FILENAME: _block(resource_count=512)})
        mod.apply_snapshot(render, {mod.AGENT_FILENAME: _block(resource_count=513)})
        text = (render / ".claude" / "agents" / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        assert "513 Resource" in text
        assert "512 Resource" not in text

    def test_attached_persona_renders_are_patched_too(self, tmp_path: Path):
        render = _render(tmp_path, "demo-readonly", "demo-readwrite")
        patched = mod.apply_snapshot(render, {mod.AGENT_FILENAME: _block()})
        assert len(patched) == 3
        for path in patched:
            assert "512 Resource" in path.read_text(encoding="utf-8")

    def test_a_hand_edited_file_is_reported_and_left_alone(self, tmp_path: Path, caplog):
        """The section survived but its markers did not: say so, never append."""
        agents = tmp_path / ".claude" / "agents"
        agents.mkdir(parents=True)
        target = agents / mod.AGENT_FILENAME
        edited = f"# Hand-edited prompt\n\n{mod.SNAPSHOT_HEADING}\n\nMarkers removed.\n"
        target.write_text(edited, encoding="utf-8")

        with caplog.at_level("WARNING"):
            patched = mod.apply_snapshot(tmp_path, {mod.AGENT_FILENAME: _block()})

        assert patched == []
        assert target.read_text(encoding="utf-8") == edited
        assert [r for r in caplog.records if r.levelname == "WARNING"], "no warning was logged"

    def test_a_render_without_the_agent_is_a_quiet_no_op(self, tmp_path: Path):
        assert mod.apply_snapshot(tmp_path, {mod.AGENT_FILENAME: _block()}) == []

    def test_image_contexts_are_patched_too(self, tmp_path: Path):
        """The bake must reach the renders the container images are built from.

        ``osprey build`` stages ``build/.image/<name>/build/`` as each image's
        build context; the seed-time bake ran after that staging and patched
        only the host renders, so every container shipped the placeholder —
        the failure the deployed prompt in this bug showed.
        """
        images = ("demo", "demo-readwrite")
        render = _render(tmp_path, "demo-readwrite", images=images)

        patched = mod.apply_snapshot(render, {mod.AGENT_FILENAME: _block()})

        assert len(patched) == 4
        for name in images:
            context = tmp_path / IMAGE_DIR_NAME / name / BUILD_DIR_NAME / ".claude" / "agents"
            assert "512 Resource" in (context / mod.AGENT_FILENAME).read_text(encoding="utf-8")

    def test_each_agent_file_gets_its_own_block(self, tmp_path: Path):
        """The channel finder's graph arm carries the same region with its own catalogue."""
        render = _render(tmp_path)
        cf = tmp_path / ".claude" / "agents" / mod.CHANNEL_FINDER_FILENAME
        cf.write_text(PLACEHOLDER.replace("facility-knowledge-graph", "channel-finder"))

        patched = mod.apply_snapshot(
            render,
            {
                mod.AGENT_FILENAME: _block(resource_count=512),
                mod.CHANNEL_FINDER_FILENAME: _block(resource_count=777),
            },
        )

        assert sorted(patched) == sorted([render / ".claude" / "agents" / mod.AGENT_FILENAME, cf])
        kg = (render / ".claude" / "agents" / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        assert "512 Resource" in kg and "777 Resource" not in kg
        assert "777 Resource" in cf.read_text(encoding="utf-8")

    def test_a_channel_finder_without_markers_is_quietly_skipped(self, tmp_path: Path, caplog):
        """A non-graph channel finder has no region; that is expected, not a warning."""
        render = _render(tmp_path)
        cf = tmp_path / ".claude" / "agents" / mod.CHANNEL_FINDER_FILENAME
        cf.write_text("---\nname: channel-finder\n---\n\n# Hierarchical arm\n", encoding="utf-8")

        with caplog.at_level("WARNING"):
            patched = mod.apply_snapshot(
                render, {mod.AGENT_FILENAME: _block(), mod.CHANNEL_FINDER_FILENAME: _block()}
            )

        assert patched == [render / ".claude" / "agents" / mod.AGENT_FILENAME]
        assert cf.read_text(encoding="utf-8").endswith("# Hierarchical arm\n")
        assert not [r for r in caplog.records if r.levelname == "WARNING"]


class TestBakeSnapshot:
    """The bake hands each agent file the catalogue its server actually serves."""

    def test_each_file_carries_its_servers_catalogue(self, tmp_path: Path, monkeypatch):
        from osprey.mcp_server.channel_finder_graph.tools.examples_data import (
            EXAMPLE_QUERIES as cf_examples,
        )
        from osprey.mcp_server.graph.tools.examples_data import EXAMPLE_QUERIES as graph_examples
        from osprey.services.facility_knowledge.seeder import graph_seeder

        monkeypatch.setattr(graph_seeder, "read_marker", lambda session: "0123456789abcdef")
        monkeypatch.setattr(graph_seeder, "resource_count", lambda session: 42)
        monkeypatch.setattr(graph_seeder, "read_direction_source", lambda session: "limits")

        agents = _render(tmp_path) / ".claude" / "agents"
        (agents / mod.CHANNEL_FINDER_FILENAME).write_text(PLACEHOLDER, encoding="utf-8")

        store = _FakeStore()
        patched = mod.bake_snapshot(_session(store), tmp_path)

        assert len(patched) == 2
        kg = (agents / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        cf = (agents / mod.CHANNEL_FINDER_FILENAME).read_text(encoding="utf-8")
        graph_only = {e.key for e in graph_examples} - {e.key for e in cf_examples}
        cf_only = {e.key for e in cf_examples} - {e.key for e in graph_examples}
        assert graph_only and cf_only, "fixture precondition: the catalogues differ"
        for key in graph_only:
            assert f"#### {key} " in kg and f"#### {key} " not in cf
        for key in cf_only:
            assert f"#### {key} " in cf and f"#### {key} " not in kg
        assert "42 Resource" in kg and "42 Resource" in cf
        for text in (kg, cf):
            assert "### Vocabulary" in text, "the store's synonyms reached only one agent"
            assert "| `VacuumGauge` |" in text
            assert "Direction provenance:" in text

        # The vocabulary and the specimen are facts about the store, not about a
        # catalogue: capturing them once and sharing them is what keeps the two
        # renders consistent — and keeps a second agent from doubling the cost.
        for cypher in (mod.VOCABULARY_CYPHER, mod.SPECIMEN_VALUES_CYPHER):
            assert [c for c, _p in store.queries if c == cypher] == [cypher]

    def test_a_store_that_answers_only_the_schema_still_bakes(self, tmp_path: Path, monkeypatch):
        """`osprey up` stages this bake: a partial store degrades, never raises."""
        from osprey.services.facility_knowledge.seeder import graph_seeder

        monkeypatch.setattr(graph_seeder, "read_marker", lambda session: None)
        monkeypatch.setattr(graph_seeder, "resource_count", lambda session: 0)
        monkeypatch.setattr(graph_seeder, "read_direction_source", lambda session: None)

        agents = _render(tmp_path) / ".claude" / "agents"
        store = _FakeStore(vocabulary=(), specimen=None)

        patched = mod.bake_snapshot(_session(store), tmp_path)

        assert patched == [agents / mod.AGENT_FILENAME]
        text = (agents / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        assert mod.NO_VOCABULARY_NOTE in text
        assert mod.DEFAULT_PARAMETERS_NOTE in text
        assert "Direction provenance" not in text

    def test_a_store_that_raises_on_both_captures_still_bakes(self, tmp_path: Path, monkeypatch):
        """The guards must hold end to end, not just around each function.

        `osprey up` stages this bake; a store erroring on the two new queries has
        to leave the deploy with a prompt rather than an exception.
        """
        from osprey.services.facility_knowledge.seeder import graph_seeder

        monkeypatch.setattr(graph_seeder, "read_marker", lambda session: "0123456789abcdef")
        monkeypatch.setattr(graph_seeder, "resource_count", lambda session: 42)
        monkeypatch.setattr(graph_seeder, "read_direction_source", lambda session: "grammar")

        agents = _render(tmp_path) / ".claude" / "agents"

        patched = mod.bake_snapshot(_session(_SchemaOnlyStore()), tmp_path)

        assert patched == [agents / mod.AGENT_FILENAME]
        text = (agents / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        assert "42 Resource" in text, "the schema capture was lost with the failing ones"
        assert mod.NO_VOCABULARY_NOTE in text
        assert mod.DEFAULT_PARAMETERS_NOTE in text
        assert mod.CAPTURED_PARAMETERS_NOTE not in text


# ---------------------------------------------------------------------------
# The template's markers are this module's markers
# ---------------------------------------------------------------------------


def test_shared_partial_carries_exactly_these_markers():
    """A marker respelled on either side turns every bake into a silent no-op.

    Every graph-querying agent includes this one partial, so pinning it pins
    them all; the channel finder's render test checks its graph arm includes it.
    """
    from osprey.cli.templates.manager import TemplateManager

    template = (
        TemplateManager().template_root
        / "claude_code"
        / "claude"
        / "agents"
        / "_shared"
        / "graph_snapshot.md.j2"
    ).read_text(encoding="utf-8")

    assert mod.SNAPSHOT_HEADING in template
    assert mod.SNAPSHOT_BEGIN in template
    assert mod.SNAPSHOT_END in template
    assert template.count("osprey:graph-snapshot") == 2, (
        "the partial must carry exactly one begin and one end marker"
    )


class TestDescribePatched:
    """The reported count is prompts × renders, not a flat file total."""

    def test_counts_distinct_prompts_and_renders(self):
        renders = [Path("build"), Path("build/persona-a"), Path("build/.image/x/build")]
        patched = [
            base / ".claude" / "agents" / name for base in renders for name in mod.TARGET_FILENAMES
        ]
        assert mod.describe_patched(patched) == "2 agent prompt(s) across 3 render(s)"

    def test_single_file(self):
        patched = [Path("build/.claude/agents") / mod.AGENT_FILENAME]
        assert mod.describe_patched(patched) == "1 agent prompt(s) across 1 render(s)"
