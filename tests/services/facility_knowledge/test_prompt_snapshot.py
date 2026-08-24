"""Unit tests for the seed-time prompt snapshot.

Three claims, each with the failure it guards against:

1. **Collection is the shared one.** ``collect_schema`` filters bookkeeping and
   skips unquotable labels exactly as the ``get_schema`` tool documents —
   asserted on the *queries issued*, not just the payload, because a collection
   that read the seed marker and filtered afterwards would leak on a store
   whose bookkeeping moves.
2. **The block is honest about its corpus.** Parameter sets are narrowed to the
   corpus the store itself names, and only when the store names exactly one the
   examples know.
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from osprey.services.facility_knowledge.seeder import prompt_snapshot as mod

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


@dataclass
class _FakeStore:
    """Routes on the Cypher text and records every query, like the tool's fake."""

    labels: tuple[str, ...] = _DEFAULT_LABELS
    relationship_types: tuple[str, ...] = _DEFAULT_RELTYPES
    keys_by_label: dict[str, list[str]] = field(default_factory=lambda: dict(_DEFAULT_KEYS))
    facilities: tuple[str, ...] = ("demo",)
    queries: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)

    def run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((cypher, params))
        if "db.labels" in cypher:
            return [{"label": name} for name in self.labels]
        if "db.relationshipTypes" in cypher:
            return [{"relationshipType": name} for name in self.relationship_types]
        if "d.facility" in cypher:
            return [{"facility": name} for name in self.facilities]
        if "UNWIND keys(n)" in cypher:
            label = cypher.split("`")[1]
            return [{"key": key} for key in self.keys_by_label.get(label, [])]
        raise AssertionError(f"unexpected query: {cypher!r}")


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
# detect_corpus
# ---------------------------------------------------------------------------


class TestDetectCorpus:
    KNOWN = frozenset({"als", "demo"})

    def test_single_known_facility_selects_its_corpus(self):
        store = _FakeStore(facilities=("demo",))
        assert mod.detect_corpus(store.run, self.KNOWN) == "demo"

    def test_facility_case_is_folded_in_cypher(self):
        """The ALS corpus spells its facility "ALS"; the Cypher lowercases it,
        so the fake — which routes, not folds — is fed the folded value."""
        store = _FakeStore(facilities=("als",))
        assert mod.detect_corpus(store.run, self.KNOWN) == "als"

    def test_unknown_facility_keeps_every_parameter_set(self):
        store = _FakeStore(facilities=("bessy",))
        assert mod.detect_corpus(store.run, self.KNOWN) is None

    def test_multiple_facilities_keep_every_parameter_set(self):
        store = _FakeStore(facilities=("als", "demo"))
        assert mod.detect_corpus(store.run, self.KNOWN) is None

    def test_no_facility_keeps_every_parameter_set(self):
        store = _FakeStore(facilities=())
        assert mod.detect_corpus(store.run, self.KNOWN) is None


# ---------------------------------------------------------------------------
# render_block / apply_snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Example:
    key: str = "q1a"
    title: str = "Device count by concrete class"
    description: str = "How many devices of each concrete kind the graph holds."
    cypher: str = "MATCH (d:Resource) RETURN count(d) LIMIT 100"
    parameters: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {"als": {"x": 1}, "demo": {"x": 2}}
    )


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
        "corpus": "demo",
        "digest": "abcdef0123456789",
        "resource_count": 512,
    }
    kwargs.update(overrides)
    return mod.render_block(_schema(), [_Example()], **kwargs)


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

    def test_detected_corpus_narrows_the_parameters(self):
        block = _block(corpus="demo")
        assert '{"x": 2}' in block
        assert '{"x": 1}' not in block, "the other corpus's values leaked into the block"

    def test_undetected_corpus_keeps_every_set_with_the_pick_note(self):
        block = _block(corpus=None)
        assert '{"x": 1}' in block
        assert '{"x": 2}' in block
        assert "pick the set matching the seeded corpus" in block

    def test_schema_and_example_render(self):
        block = _block()
        assert "HASBINDING" in block
        assert "```cypher" in block
        assert "MATCH (d:Resource) RETURN count(d) LIMIT 100" in block


class TestApplySnapshot:
    PLACEHOLDER = (
        "---\nname: facility-knowledge-graph\n---\n\n# Agent\n\n## The Graph at Hand\n\n"
        f"{mod.SNAPSHOT_BEGIN}\n\nNo snapshot yet.\n\n{mod.SNAPSHOT_END}\n\n## Submitting\n"
    )

    def _render(self, tmp_path: Path, *personas: str) -> Path:
        agents = tmp_path / ".claude" / "agents"
        agents.mkdir(parents=True)
        (agents / mod.AGENT_FILENAME).write_text(self.PLACEHOLDER, encoding="utf-8")
        for persona in personas:
            sub = tmp_path / persona / ".claude" / "agents"
            sub.mkdir(parents=True)
            (sub / mod.AGENT_FILENAME).write_text(self.PLACEHOLDER, encoding="utf-8")
        return tmp_path

    def test_replaces_the_managed_region_only(self, tmp_path: Path):
        render = self._render(tmp_path)
        patched = mod.apply_snapshot(render, _block())

        text = (render / ".claude" / "agents" / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        assert patched == [render / ".claude" / "agents" / mod.AGENT_FILENAME]
        assert "No snapshot yet." not in text
        assert "512 Resource" in text
        assert text.startswith("---\nname: facility-knowledge-graph"), "frontmatter was touched"
        assert text.endswith("## Submitting\n"), "text after the region was touched"

    def test_applying_twice_is_applying_once(self, tmp_path: Path):
        render = self._render(tmp_path)
        mod.apply_snapshot(render, _block())
        once = (render / ".claude" / "agents" / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        mod.apply_snapshot(render, _block())
        twice = (render / ".claude" / "agents" / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        assert once == twice

    def test_a_new_capture_replaces_the_old_one(self, tmp_path: Path):
        render = self._render(tmp_path)
        mod.apply_snapshot(render, _block(resource_count=512))
        mod.apply_snapshot(render, _block(resource_count=513))
        text = (render / ".claude" / "agents" / mod.AGENT_FILENAME).read_text(encoding="utf-8")
        assert "513 Resource" in text
        assert "512 Resource" not in text

    def test_attached_persona_renders_are_patched_too(self, tmp_path: Path):
        render = self._render(tmp_path, "demo-readonly", "demo-readwrite")
        patched = mod.apply_snapshot(render, _block())
        assert len(patched) == 3
        for path in patched:
            assert "512 Resource" in path.read_text(encoding="utf-8")

    def test_a_file_without_markers_is_left_alone(self, tmp_path: Path):
        agents = tmp_path / ".claude" / "agents"
        agents.mkdir(parents=True)
        target = agents / mod.AGENT_FILENAME
        target.write_text("# Hand-edited prompt, markers removed\n", encoding="utf-8")

        patched = mod.apply_snapshot(tmp_path, _block())
        assert patched == []
        assert target.read_text(encoding="utf-8") == "# Hand-edited prompt, markers removed\n"

    def test_a_render_without_the_agent_is_a_quiet_no_op(self, tmp_path: Path):
        assert mod.apply_snapshot(tmp_path, _block()) == []


# ---------------------------------------------------------------------------
# The template's markers are this module's markers
# ---------------------------------------------------------------------------


def test_agent_template_carries_exactly_these_markers():
    """A marker respelled on either side turns every bake into a silent no-op."""
    from osprey.cli.templates.manager import TemplateManager

    template = (
        TemplateManager().template_root
        / "claude_code"
        / "claude"
        / "agents"
        / "facility-knowledge-graph.md.j2"
    ).read_text(encoding="utf-8")

    assert mod.SNAPSHOT_BEGIN in template
    assert mod.SNAPSHOT_END in template
    assert template.count("osprey:graph-snapshot") == 2, (
        "the template must carry exactly one begin and one end marker"
    )
