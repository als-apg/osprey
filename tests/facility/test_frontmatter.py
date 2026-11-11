"""The facility layer's Markdown-with-frontmatter reader."""

from __future__ import annotations

import pytest

from osprey.facility.frontmatter import FrontmatterError, split_frontmatter
from tests._fresh_import import modules_added_by_import


def test_text_without_a_leading_delimiter_is_all_body() -> None:
    text = "# Title\n\n---\nnot: frontmatter\n---\n"
    assert split_frontmatter(text) == ({}, text)


def test_empty_text_is_all_body() -> None:
    assert split_frontmatter("") == ({}, "")


def test_frontmatter_and_body_split() -> None:
    frontmatter, body = split_frontmatter("---\ndevice_id: BPM1\n---\n\n# Page\n")
    assert frontmatter == {"device_id": "BPM1"}
    assert body == "# Page"


def test_body_drops_exactly_one_leading_blank_line() -> None:
    _, body = split_frontmatter("---\na: 1\n---\n\n\n# Page")
    assert body == "\n# Page"


def test_empty_block_is_an_empty_mapping() -> None:
    assert split_frontmatter("---\n---\nbody") == ({}, "body")


def test_unterminated_block_raises() -> None:
    with pytest.raises(FrontmatterError, match="^Unterminated YAML frontmatter block$"):
        split_frontmatter("---\na: 1\n")


def test_invalid_yaml_raises() -> None:
    with pytest.raises(FrontmatterError, match="^Invalid YAML in frontmatter: "):
        split_frontmatter("---\na: [1\n---\n")


def test_non_mapping_yaml_raises() -> None:
    with pytest.raises(FrontmatterError, match="^Frontmatter must be a YAML mapping$"):
        split_frontmatter("---\n- a\n- b\n---\n")


def test_error_is_a_value_error() -> None:
    assert issubclass(FrontmatterError, ValueError)


def test_facility_build_does_not_import_knowledge() -> None:
    added = modules_added_by_import("osprey.facility.build")
    assert not any(
        name == "osprey.services.facility_knowledge"
        or name.startswith("osprey.services.facility_knowledge.")
        for name in added
    )
