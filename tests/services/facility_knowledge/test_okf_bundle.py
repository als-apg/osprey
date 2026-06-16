"""Tests for OKFBundle: list_concepts, read_concept, and search."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from osprey.services.facility_knowledge.okf.bundle import (
    ConceptEntry,
    OKFBundle,
    OKFBundleError,
)
from osprey.services.facility_knowledge.okf.document import OKFDocument, OKFDocumentError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(body).lstrip(), encoding="utf-8")


def _make_bundle(root: Path) -> None:
    """Populate a minimal bundle used across multiple tests."""
    _write(
        root / "index.md",
        """\
        # Facility Bundle

        * [Accelerator Overview](/accelerator_overview.md) - Top-level overview.
        * [Tables](/tables/) - Data tables.
        """,
    )
    _write(
        root / "accelerator_overview.md",
        """\
        ---
        type: facility_overview
        title: Accelerator Overview
        description: Overview of the accelerator complex.
        ---

        The accelerator uses a synchrotron ring to produce light.
        """,
    )
    _write(
        root / "tables" / "index.md",
        """\
        # Tables

        * [Beam Parameters](/tables/beam_params.md) - Beam parameter table.
        """,
    )
    _write(
        root / "tables" / "beam_params.md",
        """\
        ---
        type: data_table
        title: Beam Parameters
        description: Live beam parameter values from EPICS.
        tags: [epics, beam]
        ---

        Contains PV names for all beam parameters.
        """,
    )
    _write(
        root / "tables" / "broken_links.md",
        """\
        ---
        type: data_table
        title: Table With Broken Link
        description: References a nonexistent doc.
        ---

        See [missing concept](/tables/does_not_exist.md) for details.
        """,
    )


# ---------------------------------------------------------------------------
# OKFBundle construction
# ---------------------------------------------------------------------------


class TestOKFBundleConstruction:
    """Bundle raises on missing root."""

    def test_raises_when_root_missing(self, tmp_path: Path):
        with pytest.raises(OKFBundleError, match="does not exist"):
            OKFBundle(tmp_path / "no_such_dir")

    def test_accepts_existing_root(self, tmp_path: Path):
        bundle = OKFBundle(tmp_path)
        assert bundle.root == tmp_path

    def test_root_is_coerced_to_path(self, tmp_path: Path):
        bundle = OKFBundle(str(tmp_path))  # type: ignore[arg-type]
        assert isinstance(bundle.root, Path)


# ---------------------------------------------------------------------------
# list_concepts
# ---------------------------------------------------------------------------


class TestListConcepts:
    """list_concepts reads index.md without loading full doc bodies."""

    def test_returns_entries_for_all_concepts(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        entries = bundle.list_concepts()

        ids = {e.concept_id for e in entries}
        assert "accelerator_overview" in ids
        assert "tables/beam_params" in ids
        assert "tables/broken_links" in ids

    def test_index_md_is_not_returned_as_concept(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        entries = bundle.list_concepts()

        ids = {e.concept_id for e in entries}
        assert "index" not in ids
        assert "tables/index" not in ids

    def test_returns_concept_entry_type(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        entries = bundle.list_concepts()

        for e in entries:
            assert isinstance(e, ConceptEntry)
            assert e.concept_id
            assert e.title

    def test_index_entries_include_title_and_description(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        entries = bundle.list_concepts()

        by_id = {e.concept_id: e for e in entries}
        overview = by_id.get("accelerator_overview")
        assert overview is not None
        assert overview.title == "Accelerator Overview"
        assert "overview" in overview.description.lower()

    def test_empty_bundle_returns_no_entries(self, tmp_path: Path):
        bundle = OKFBundle(tmp_path)
        assert bundle.list_concepts() == []

    def test_fallback_scan_covers_docs_not_in_index(self, tmp_path: Path):
        """Concepts absent from index.md are still returned via filesystem scan."""
        _write(
            tmp_path / "orphan.md",
            """\
            ---
            type: note
            title: Orphan
            description: Not listed in any index.
            ---
            Body text.
            """,
        )
        bundle = OKFBundle(tmp_path)
        ids = {e.concept_id for e in bundle.list_concepts()}
        assert "orphan" in ids

    def test_no_duplicate_entries(self, tmp_path: Path):
        """A doc listed in index.md must not appear twice."""
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        entries = bundle.list_concepts()

        ids = [e.concept_id for e in entries]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# read_concept
# ---------------------------------------------------------------------------


class TestReadConcept:
    """read_concept loads a document by its OKF §2 concept ID."""

    def test_reads_existing_concept(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        doc = bundle.read_concept("accelerator_overview")

        assert isinstance(doc, OKFDocument)
        assert doc.frontmatter["type"] == "facility_overview"
        assert "synchrotron" in doc.body

    def test_reads_nested_concept(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        doc = bundle.read_concept("tables/beam_params")

        assert doc.frontmatter["title"] == "Beam Parameters"

    def test_raises_for_missing_concept(self, tmp_path: Path):
        bundle = OKFBundle(tmp_path)

        with pytest.raises(OKFBundleError, match="Concept not found"):
            bundle.read_concept("no/such/concept")

    def test_broken_cross_links_in_body_are_tolerated(self, tmp_path: Path):
        """A doc with links to non-existent concepts must still load (OKF §9)."""
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        doc = bundle.read_concept("tables/broken_links")

        assert doc.frontmatter["title"] == "Table With Broken Link"
        assert "does_not_exist" in doc.body

    def test_raises_okf_document_error_for_bad_frontmatter(self, tmp_path: Path):
        _write(
            tmp_path / "bad.md",
            "---\n- not\n- a\n- mapping\n---\n\nBody.\n",
        )
        bundle = OKFBundle(tmp_path)

        with pytest.raises(OKFDocumentError):
            bundle.read_concept("bad")

    def test_concept_id_with_leading_slash_is_tolerated(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        doc = bundle.read_concept("/accelerator_overview")

        assert doc.frontmatter["type"] == "facility_overview"

    # --- path-traversal containment (security regression) ---

    def test_dotdot_traversal_raises(self, tmp_path: Path):
        """concept_id with .. must raise OKFBundleError, not read outside root."""
        secret = tmp_path.parent / "secret.md"
        secret.write_text("---\ntype: secret\n---\nshould not be readable\n", encoding="utf-8")

        bundle_root = tmp_path / "bundle"
        bundle_root.mkdir()
        bundle = OKFBundle(bundle_root)

        with pytest.raises(OKFBundleError, match="escapes bundle root"):
            bundle.read_concept("../secret")

    def test_leading_slash_dotdot_traversal_raises(self, tmp_path: Path):
        """concept_id '/../secret' must also be blocked."""
        secret = tmp_path.parent / "secret2.md"
        secret.write_text("---\ntype: secret\n---\nshould not be readable\n", encoding="utf-8")

        bundle_root = tmp_path / "bundle"
        bundle_root.mkdir()
        bundle = OKFBundle(bundle_root)

        with pytest.raises(OKFBundleError, match="escapes bundle root"):
            bundle.read_concept("/../secret2")

    def test_double_slash_dotdot_traversal_raises(self, tmp_path: Path):
        """Embedded // + .. traversal must be caught after path resolution."""
        secret = tmp_path.parent / "secret3.md"
        secret.write_text("---\ntype: secret\n---\nshould not be readable\n", encoding="utf-8")

        bundle_root = tmp_path / "bundle"
        bundle_root.mkdir()
        bundle = OKFBundle(bundle_root)

        # concept_id that, after resolve(), points outside the root.
        with pytest.raises(OKFBundleError, match="escapes bundle root"):
            bundle.read_concept("subdir/../../secret3")

    def test_normal_nested_concept_still_works_after_containment_fix(self, tmp_path: Path):
        """Legitimate nested IDs must still resolve correctly."""
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        doc = bundle.read_concept("tables/beam_params")

        assert doc.frontmatter["title"] == "Beam Parameters"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    """search matches across frontmatter values and body text."""

    def test_search_matches_body_text(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        results = bundle.search("synchrotron")

        ids = {cid for cid, _ in results}
        assert "accelerator_overview" in ids

    def test_search_matches_frontmatter_value(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        results = bundle.search("epics")

        ids = {cid for cid, _ in results}
        assert "tables/beam_params" in ids

    def test_search_is_case_insensitive(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)

        lower = {cid for cid, _ in bundle.search("synchrotron")}
        upper = {cid for cid, _ in bundle.search("SYNCHROTRON")}
        assert lower == upper

    def test_search_returns_okf_documents(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        results = bundle.search("beam")

        assert all(isinstance(doc, OKFDocument) for _, doc in results)

    def test_search_no_match_returns_empty_list(self, tmp_path: Path):
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        assert bundle.search("zyxwvutsrqponmlkjihgfedcba") == []

    def test_search_does_not_return_index_md(self, tmp_path: Path):
        """index.md files must never appear in search results."""
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        # The index contains the word "Facility" in a heading.
        results = bundle.search("Facility")

        ids = {cid for cid, _ in results}
        assert "index" not in ids
        assert "tables/index" not in ids

    def test_search_matches_tag_in_frontmatter_list(self, tmp_path: Path):
        """Tags are a YAML list; search must reach into them."""
        _make_bundle(tmp_path)
        bundle = OKFBundle(tmp_path)
        results = bundle.search("beam")

        ids = {cid for cid, _ in results}
        assert "tables/beam_params" in ids
