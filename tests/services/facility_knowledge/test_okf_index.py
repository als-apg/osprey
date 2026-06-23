"""Tests for OKF index regeneration and validation.

Covers:
- regenerate_indexes: one index.md per non-empty directory, grouped by type,
  bundle-relative /… links, root index gets okf_version frontmatter.
- _synthesize_description: deterministic, no model call.
- validate_index: okf_version allowed in root index.md, forbidden elsewhere;
  any non-root index.md with frontmatter raises OKFIndexError.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from osprey.services.facility_knowledge.okf.index import (
    OKFIndexError,
    _synthesize_description,
    regenerate_indexes,
    validate_index,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _concept(tmp_path: Path, rel: str, *, type_: str, title: str, desc: str = "") -> Path:
    """Write a minimal concept document into the fixture bundle."""
    p = tmp_path / rel
    fm_lines = [f"type: {type_}", f"title: {title}"]
    if desc:
        fm_lines.append(f"description: {desc}")
    content = "---\n" + "\n".join(fm_lines) + "\n---\n\nBody text.\n"
    _write(p, content)
    return p


# ---------------------------------------------------------------------------
# _synthesize_description (deterministic; no model)
# ---------------------------------------------------------------------------


class TestSynthesizeDescription:
    def test_empty_children_returns_empty(self):
        assert _synthesize_description([]) == ""

    def test_single_child_with_description_returns_description(self):
        result = _synthesize_description([("Title A", "A helpful description.")])
        assert result == "A helpful description."

    def test_single_child_no_description_returns_contains_line(self):
        result = _synthesize_description([("Title A", "")])
        assert "Title A" in result
        assert "1 entries" in result

    def test_multiple_children_returns_contains_line(self):
        result = _synthesize_description([("Alpha", "desc"), ("Beta", ""), ("Gamma", "")])
        assert "3 entries" in result
        assert "Alpha" in result
        assert "Beta" in result

    def test_no_model_call(self):
        """Verify determinism: calling twice returns identical output."""
        children = [("X", ""), ("Y", "Ydesc")]
        assert _synthesize_description(children) == _synthesize_description(children)


# ---------------------------------------------------------------------------
# regenerate_indexes — core behaviour
# ---------------------------------------------------------------------------


class TestRegenerateIndexes:
    def test_nonexistent_root_returns_empty(self, tmp_path):
        result = regenerate_indexes(tmp_path / "does_not_exist")
        assert result == []

    def test_single_concept_at_root(self, tmp_path):
        """Single concept → root index.md is written."""
        _concept(tmp_path, "overview.md", type_="facility_overview", title="Overview")

        written = regenerate_indexes(tmp_path)

        assert len(written) == 1
        assert written[0] == tmp_path / "index.md"

    def test_root_index_has_okf_version_frontmatter(self, tmp_path):
        """Root index.md must include okf_version frontmatter (OKF §11)."""
        _concept(tmp_path, "a.md", type_="Device", title="Device A")

        regenerate_indexes(tmp_path)

        text = (tmp_path / "index.md").read_text()
        assert text.startswith("---\n")
        assert "okf_version" in text

    def test_sub_index_has_no_frontmatter(self, tmp_path):
        """Sub-directory index.md must NOT contain frontmatter (OKF §6)."""
        _concept(tmp_path, "devices/pump.md", type_="Device", title="Pump")

        regenerate_indexes(tmp_path)

        sub_index = tmp_path / "devices" / "index.md"
        assert sub_index.exists()
        text = sub_index.read_text()
        assert not text.startswith("---\n")

    def test_entries_grouped_by_type(self, tmp_path):
        """Concepts of different types appear under their own section headings."""
        _concept(tmp_path, "device.md", type_="Device", title="Ion Pump")
        _concept(tmp_path, "proc.md", type_="Procedure", title="Startup")

        regenerate_indexes(tmp_path)

        text = (tmp_path / "index.md").read_text()
        assert "# Device" in text
        assert "# Procedure" in text

    def test_links_are_bundle_relative(self, tmp_path):
        """All links in index.md start with /."""
        _concept(tmp_path, "subsys/magnet.md", type_="Device", title="Magnet")

        regenerate_indexes(tmp_path)

        sub_text = (tmp_path / "subsys" / "index.md").read_text()
        assert "[Magnet](/subsys/magnet.md)" in sub_text

    def test_description_included_in_entry(self, tmp_path):
        """Concept descriptions appear as suffixes in the index entry."""
        _concept(
            tmp_path,
            "vac.md",
            type_="System",
            title="Vacuum System",
            desc="Ultra-high vacuum across all beamlines.",
        )

        regenerate_indexes(tmp_path)

        text = (tmp_path / "index.md").read_text()
        assert "Ultra-high vacuum across all beamlines." in text

    def test_subdirectory_entry_appears_in_parent_index(self, tmp_path):
        """Sub-directories are listed as Subdirectories entries in parent index."""
        _concept(tmp_path, "devices/pump.md", type_="Device", title="Pump")

        regenerate_indexes(tmp_path)

        root_text = (tmp_path / "index.md").read_text()
        assert "# Subdirectories" in root_text
        assert "[devices](/devices/)" in root_text

    def test_empty_directory_not_indexed(self, tmp_path):
        """Directories that only contain an index.md (no concepts) are skipped."""
        # Create a directory with no concepts
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        # Root has one concept so a root index is written
        _concept(tmp_path, "root.md", type_="X", title="Root Concept")

        regenerate_indexes(tmp_path)

        assert not (empty_dir / "index.md").exists()

    def test_existing_index_not_included_as_concept(self, tmp_path):
        """Pre-existing index.md files are skipped, not indexed as concepts."""
        _concept(tmp_path, "a.md", type_="Device", title="A")
        # Pre-seed an index.md that would confuse the indexer if treated as concept
        (tmp_path / "index.md").write_text("# Old index\n", encoding="utf-8")

        regenerate_indexes(tmp_path)

        text = (tmp_path / "index.md").read_text()
        # Should contain the real concept, not a self-referential index entry
        assert "[A]" in text
        assert "[index]" not in text

    def test_returns_written_paths(self, tmp_path):
        """Return value lists every index.md that was written."""
        _concept(tmp_path, "sub/c.md", type_="T", title="C")

        written = regenerate_indexes(tmp_path)

        written_names = {p.name for p in written}
        assert written_names == {"index.md"}
        written_dirs = {p.parent for p in written}
        assert tmp_path in written_dirs
        assert tmp_path / "sub" in written_dirs

    def test_unparseable_file_is_skipped(self, tmp_path):
        """A .md file with broken frontmatter is silently skipped."""
        bad = tmp_path / "broken.md"
        bad.write_text("---\nkey: [\nunclosed\n---\nBody\n", encoding="utf-8")
        _concept(tmp_path, "good.md", type_="Device", title="Good")

        regenerate_indexes(tmp_path)

        text = (tmp_path / "index.md").read_text()
        assert "[Good]" in text
        # broken.md should not appear
        assert "[broken]" not in text

    def test_no_network_or_model_call(self, tmp_path, monkeypatch):
        """regenerate_indexes must not import or call any LLM/network library."""
        import sys

        # Block any import of google.genai or anthropic at the module level
        for blocked in ("google", "anthropic", "openai"):
            monkeypatch.setitem(sys.modules, blocked, None)  # type: ignore[arg-type]

        # Should complete without raising ImportError for the blocked modules
        _concept(tmp_path, "sub/x.md", type_="T", title="X", desc="A description.")
        regenerate_indexes(tmp_path)  # must not raise


# ---------------------------------------------------------------------------
# validate_index
# ---------------------------------------------------------------------------


class TestValidateIndex:
    def test_root_index_with_okf_version_passes(self, tmp_path):
        idx = tmp_path / "index.md"
        idx.write_text(
            '---\nokf_version: "0.1"\n---\n\n# Device\n\n* [X](/x.md)\n', encoding="utf-8"
        )

        validate_index(idx, bundle_root=tmp_path)  # must not raise

    def test_root_index_no_frontmatter_passes(self, tmp_path):
        idx = tmp_path / "index.md"
        idx.write_text("# Device\n\n* [X](/x.md)\n", encoding="utf-8")

        validate_index(idx, bundle_root=tmp_path)  # must not raise

    def test_root_index_extra_key_raises(self, tmp_path):
        idx = tmp_path / "index.md"
        idx.write_text('---\nokf_version: "0.1"\ntitle: Root\n---\n\n# Device\n', encoding="utf-8")

        with pytest.raises(OKFIndexError, match="title"):
            validate_index(idx, bundle_root=tmp_path)

    def test_sub_index_no_frontmatter_passes(self, tmp_path):
        sub = tmp_path / "devices"
        sub.mkdir()
        idx = sub / "index.md"
        idx.write_text("# Device\n\n* [Pump](/devices/pump.md)\n", encoding="utf-8")

        validate_index(idx, bundle_root=tmp_path)  # must not raise

    def test_sub_index_with_frontmatter_raises(self, tmp_path):
        sub = tmp_path / "devices"
        sub.mkdir()
        idx = sub / "index.md"
        idx.write_text('---\nokf_version: "0.1"\n---\n\n# Device\n', encoding="utf-8")

        with pytest.raises(OKFIndexError, match="sub-directory"):
            validate_index(idx, bundle_root=tmp_path)

    def test_regenerated_indexes_all_pass_validation(self, tmp_path):
        """Every index.md produced by regenerate_indexes passes validate_index."""
        _concept(
            tmp_path, "systems/vacuum.md", type_="System", title="Vacuum", desc="Vacuum system."
        )
        _concept(tmp_path, "systems/timing.md", type_="System", title="Timing")
        _concept(tmp_path, "devices/pump.md", type_="Device", title="Ion Pump")

        written = regenerate_indexes(tmp_path)

        for idx_path in written:
            validate_index(idx_path, bundle_root=tmp_path)  # must not raise
