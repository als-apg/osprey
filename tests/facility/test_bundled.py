"""``osprey.facility.bundled``: what a repo's facility tree holds, by origin."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from osprey.facility.bundled import HEADER_CAPABLE, bundled_origin, classify, header_line
from osprey.facility.layers import HEADER
from osprey.facility.layers.mml.mapping import MAPPING_FILE
from tests._builds import init_project

TEMPLATE_ROOT = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"
FACILITIES = TEMPLATE_ROOT / "facilities"


def _is_index(relative: Path) -> bool:
    return relative.parts[0] == "knowledge" and relative.name == "index.md"


def _states(path: Path, line: str) -> bool:
    lines = path.read_text(encoding="utf-8").split("\n")
    if path.suffix == ".md" and lines[0] == "---":
        return lines[1] == line
    if path.suffix == ".md":
        return lines[0] == f"<!-- {line.removeprefix('# ')} -->"
    return lines[0] == line


def test_every_packaged_text_file_states_its_origin() -> None:
    checked = 0
    for name in ("example", "hello_world"):
        tree = FACILITIES / name
        for path in sorted(tree.rglob("*")):
            relative = path.relative_to(tree)
            if path.is_file() and path.suffix in HEADER_CAPABLE and not _is_index(relative):
                checked += 1
                assert _states(path, header_line(name)), relative
                assert bundled_origin(path) == name, relative
    assert checked


def test_no_knowledge_index_carries_the_line() -> None:
    indexes = sorted((FACILITIES / "example" / "knowledge").rglob("index.md"))
    assert indexes
    assert [path for path in indexes if bundled_origin(path) is not None] == []


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return init_project(tmp_path_factory.mktemp("ca"), "demo", "demo")


@pytest.fixture
def facility(initialised: Path, tmp_path: Path) -> Path:
    copy = tmp_path / initialised.name
    shutil.copytree(initialised, copy, symlinks=True)
    return copy / "data" / "facility"


def _drop_header(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    del lines[1 if lines[0] == "---" else 0]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_a_fresh_init_is_wholly_bundled(facility: Path) -> None:
    kinds = classify(facility, TEMPLATE_ROOT)
    tree = FACILITIES / "example"
    packaged = [
        path for path in tree.rglob("*") if path.is_file() and not _is_index(path.relative_to(tree))
    ]

    assert kinds.wholly_bundled
    assert kinds.name == "example"
    assert len(kinds.bundled) == len(packaged)


def test_regenerated_indexes_leave_the_tree_wholly_bundled(facility: Path) -> None:
    from osprey.services.facility_knowledge.okf.index import regenerate_indexes

    regenerate_indexes(facility / "knowledge")

    assert classify(facility, TEMPLATE_ROOT).wholly_bundled


def test_a_reviewed_mapping_leaves_the_tree_wholly_bundled(facility: Path) -> None:
    mapping = facility / MAPPING_FILE
    mapping.parent.mkdir(parents=True)
    mapping.write_text("models: {}\n", encoding="utf-8")

    assert classify(facility, TEMPLATE_ROOT).wholly_bundled


def test_an_added_record_file_is_authored(facility: Path) -> None:
    (facility / "records" / "x.yaml").write_text("[]\n", encoding="utf-8")

    kinds = classify(facility, TEMPLATE_ROOT)

    assert kinds.authored == ("records/x.yaml",)
    assert not kinds.wholly_bundled


def test_a_changed_deck_is_authored(facility: Path) -> None:
    deck = sorted((facility / "decks").iterdir())[0]
    deck.write_bytes(deck.read_bytes() + b" ")

    assert classify(facility, TEMPLATE_ROOT).authored == (f"decks/{deck.name}",)


def test_an_edit_that_keeps_the_line_is_still_bundled(facility: Path) -> None:
    path = facility / "limits.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")

    assert classify(facility, TEMPLATE_ROOT).wholly_bundled


def test_a_file_under_the_seed_header_is_seeded(facility: Path) -> None:
    (facility / "limits.yaml").write_text(f"{HEADER}\nrecords: []\n", encoding="utf-8")

    assert classify(facility, TEMPLATE_ROOT).seeded == ("limits.yaml",)


def test_a_markdown_file_whose_frontmatter_opens_with_the_seed_header_is_seeded(
    facility: Path,
) -> None:
    (facility / "knowledge" / "seeded.md").write_text(f"---\n{HEADER}\n---\n", encoding="utf-8")

    assert classify(facility, TEMPLATE_ROOT).seeded == ("knowledge/seeded.md",)


def test_a_page_whose_line_is_deleted_is_authored(facility: Path) -> None:
    _drop_header(facility / "knowledge" / "devices" / "bpm.md")

    assert classify(facility, TEMPLATE_ROOT).authored == ("knowledge/devices/bpm.md",)


def test_a_file_under_imported_is_a_layer_file(facility: Path) -> None:
    layer = facility / "imported" / "list" / "channels.yaml"
    layer.parent.mkdir(parents=True)
    layer.write_text("[]\n", encoding="utf-8")

    kinds = classify(facility, TEMPLATE_ROOT)

    assert kinds.layer == ("imported/list/channels.yaml",)
    assert not kinds.wholly_bundled


def test_two_names_name_no_facility_and_every_file_is_authored(facility: Path) -> None:
    (facility / "other.yaml").write_text(f"{header_line('hello_world')}\n", encoding="utf-8")

    kinds = classify(facility, TEMPLATE_ROOT)

    assert kinds.name is None
    assert kinds.bundled == ()
    assert {"other.yaml", "decks/SR.json"} <= set(kinds.authored)


def test_a_comment_form_line_names_its_facility(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(f"<!-- {header_line('example')[2:]} -->\n# Page\n", encoding="utf-8")

    assert bundled_origin(page) == "example"


def test_a_missing_directory_classifies_empty(tmp_path: Path) -> None:
    kinds = classify(tmp_path / "absent", TEMPLATE_ROOT)

    assert (kinds.name, kinds.wholly_bundled) == (None, False)
