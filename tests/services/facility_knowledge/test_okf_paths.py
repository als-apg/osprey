"""Tests for the OKF file and link rules every reader of a bundle shares.

Covers OKF v0.3 §3.2 (which files are part of a bundle and which are concepts)
and §6 (one resolution rule for links and path fields).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.services.facility_knowledge.okf.document import OKFDocumentError
from osprey.services.facility_knowledge.okf.paths import (
    bundle_markdown_files,
    concept_id_of_link,
    load_concept,
    resolve_in_bundle,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


_CONCEPT = "---\ntype: Device\ntitle: Pump\n---\n\nBody.\n"


class TestBundleMarkdownFiles:
    def test_lists_md_files_at_every_depth_sorted(self, tmp_path: Path) -> None:
        _write(tmp_path / "b.md", _CONCEPT)
        _write(tmp_path / "a" / "c.md", _CONCEPT)
        _write(tmp_path / "a" / "logo.png", "not markdown")

        found = [p.relative_to(tmp_path).as_posix() for p in bundle_markdown_files(tmp_path)]

        assert found == ["a/c.md", "b.md"]

    @pytest.mark.parametrize("rel", [".hidden.md", ".git/notes.md", "a/.cache/x.md"])
    def test_dot_paths_are_not_part_of_the_bundle(self, tmp_path: Path, rel: str) -> None:
        _write(tmp_path / rel, _CONCEPT)

        assert list(bundle_markdown_files(tmp_path)) == []

    def test_a_symlink_leaving_the_root_is_not_followed(self, tmp_path: Path) -> None:
        outside = _write(tmp_path / "outside" / "secret.md", _CONCEPT)
        root = tmp_path / "bundle"
        root.mkdir()
        (root / "leak.md").symlink_to(outside)
        _write(root / "inside.md", _CONCEPT)
        (root / "alias.md").symlink_to(root / "inside.md")

        found = [p.name for p in bundle_markdown_files(root)]

        assert found == ["alias.md", "inside.md"]

    def test_reserved_files_are_listed(self, tmp_path: Path) -> None:
        """The walk names bundle files; which of them are concepts is the caller's question."""
        _write(tmp_path / "index.md", "# Device\n")
        _write(tmp_path / "log.md", "# Directory Update Log\n")

        assert [p.name for p in bundle_markdown_files(tmp_path)] == ["index.md", "log.md"]


class TestLoadConcept:
    def test_a_typed_page_is_a_concept(self, tmp_path: Path) -> None:
        doc = load_concept(_write(tmp_path / "pump.md", _CONCEPT))

        assert doc is not None
        assert doc.frontmatter["title"] == "Pump"

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("# Read me\n\nNo frontmatter.\n", id="no-frontmatter"),
            pytest.param("---\nname: a-skill\ndescription: x\n---\n\nBody.\n", id="no-type"),
            pytest.param("---\ntype: ''\n---\n\nBody.\n", id="empty-type"),
            pytest.param("---\ntype:\n---\n\nBody.\n", id="null-type"),
        ],
    )
    def test_a_page_without_a_type_is_not_a_concept(self, tmp_path: Path, text: str) -> None:
        assert load_concept(_write(tmp_path / "README.md", text)) is None

    @pytest.mark.parametrize("name", ["index.md", "log.md"])
    def test_a_reserved_file_is_never_a_concept(self, tmp_path: Path, name: str) -> None:
        """OKF §3.1: whatever frontmatter a reserved file holds."""
        assert load_concept(_write(tmp_path / name, _CONCEPT)) is None

    def test_broken_frontmatter_raises(self, tmp_path: Path) -> None:
        """Broken is not the same as "not a concept": the caller decides what it means."""
        path = _write(tmp_path / "broken.md", "---\nkey: [\n---\n")

        with pytest.raises(OKFDocumentError):
            load_concept(path)


class TestResolveInBundle:
    @pytest.mark.parametrize(
        ("target", "source_dir", "expected"),
        [
            pytest.param("/tables/users.md", "", "tables/users.md", id="root-form-from-root"),
            pytest.param("/tables/users.md", "a/b", "tables/users.md", id="root-form-from-nested"),
            pytest.param("other.md", "devices", "devices/other.md", id="bare-relative"),
            pytest.param("./other.md", "devices", "devices/other.md", id="dot-relative"),
            pytest.param("../tables/users.md", "devices", "tables/users.md", id="parent-relative"),
            pytest.param("other.md", "", "other.md", id="relative-from-root"),
            pytest.param("/a/b.md#section", "", "a/b.md", id="fragment-dropped"),
            pytest.param("my%20concept.md", "x", "x/my concept.md", id="percent-decoded"),
            pytest.param("/tables/", "", "tables/", id="directory"),
            pytest.param("sub/", "a", "a/sub/", id="relative-directory"),
            pytest.param("/assets/ring.png", "x", "assets/ring.png", id="asset"),
        ],
    )
    def test_in_bundle_targets(self, target: str, source_dir: str, expected: str) -> None:
        assert resolve_in_bundle(target, source_dir=source_dir) == expected

    @pytest.mark.parametrize(
        "target",
        [
            "https://example.org/x.md",
            "http://example.org",
            "mailto:ops@example.org",
            "facility:SR:BPM:01",
            "#section",
            "",
            "   ",
        ],
    )
    def test_external_targets_resolve_to_nothing(self, target: str) -> None:
        assert resolve_in_bundle(target, source_dir="devices") is None

    @pytest.mark.parametrize(
        ("target", "source_dir"),
        [("../secret.md", ""), ("../../secret.md", "devices"), ("/../secret.md", "devices")],
    )
    def test_a_target_outside_the_root_is_external(self, target: str, source_dir: str) -> None:
        """OKF §6: a value that resolves outside the bundle root is never read."""
        assert resolve_in_bundle(target, source_dir=source_dir) is None


class TestConceptIdOfLink:
    @pytest.mark.parametrize(
        ("target", "source_dir", "expected"),
        [
            ("/tables/users.md", "devices", "tables/users"),
            ("users.md", "tables", "tables/users"),
            ("../tables/users.md#schema", "devices", "tables/users"),
        ],
    )
    def test_a_link_to_a_markdown_page_names_a_concept(
        self, target: str, source_dir: str, expected: str
    ) -> None:
        assert concept_id_of_link(target, source_dir=source_dir) == expected

    @pytest.mark.parametrize(
        "target",
        ["/tables/", "/assets/ring.png", "/index.md", "/tables/log.md", "https://x.org/a.md"],
    )
    def test_other_targets_name_no_concept(self, target: str) -> None:
        assert concept_id_of_link(target, source_dir="") is None
