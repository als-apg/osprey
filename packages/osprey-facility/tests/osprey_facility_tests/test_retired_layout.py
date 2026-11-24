"""The retired-layout detector: what an old facility tree still holds.

Old trees are laid out from literal paths here, so a declaration that stops
naming a retired path fails these tests rather than silently shrinking them.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from osprey_facility.retired import Clash, DeckEdit, Move, RetiredLayout, find_retired
from osprey_facility_tests._harness import REPO_ROOT

BUNDLED_EXAMPLE = REPO_ROOT / "src/osprey/templates/facilities/example"


def _write(root: Path, relative: str, text: str = "x\n") -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_a_new_layout_tree_holds_nothing_retired(tmp_path: Path) -> None:
    _write(tmp_path, "identity.yaml")
    _write(tmp_path, "vocabulary/classes.yaml")
    _write(tmp_path, "simulation/decks/SR.json")
    _write(tmp_path, "simulation/models.yaml", "- name: SR\n  deck: simulation/decks/SR.json\n")
    _write(tmp_path, "benchmarks/channel_finder/tree_queries.json")

    found = find_retired(tmp_path)

    assert found == RetiredLayout(moves=(), deck_edits=(), clashes=())
    assert found.empty


def test_the_bundled_example_holds_nothing_retired() -> None:
    assert find_retired(BUNDLED_EXAMPLE).empty


def test_a_missing_tree_holds_nothing_retired(tmp_path: Path) -> None:
    assert find_retired(tmp_path / "absent").empty


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("classes.yaml", "vocabulary/classes.yaml"),
        ("signals.yaml", "vocabulary/signals.yaml"),
        ("documents.yaml", "provenance/documents.yaml"),
        ("policy.yaml", "provenance/policy.yaml"),
        ("fixes.yaml", "provenance/fixes.yaml"),
        ("seeds.yaml", "simulation/seeds.yaml"),
    ],
    ids=lambda value: value,
)
def test_a_retired_file_moves_whole(tmp_path: Path, old: str, new: str) -> None:
    _write(tmp_path, old)

    found = find_retired(tmp_path)

    assert found.moves == (Move(root=tmp_path, old=old, new=new, is_dir=False, merge=False),)
    assert not found.empty


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("decks", "simulation/decks"),
        ("measurement", "simulation/measurement"),
        ("scenarios", "simulation/scenarios"),
        ("benchmarks/cross_paradigm/queries", "benchmarks/channel_finder"),
    ],
    ids=lambda value: value,
)
def test_a_retired_directory_moves_whole_when_its_successor_is_absent(
    tmp_path: Path, old: str, new: str
) -> None:
    _write(tmp_path, f"{old}/a.yaml")
    _write(tmp_path, f"{old}/nested/b.yaml")

    found = find_retired(tmp_path)

    assert found.moves == (Move(root=tmp_path, old=old, new=new, is_dir=True, merge=False),)
    assert found.clashes == ()


def test_moves_follow_declaration_order(tmp_path: Path) -> None:
    for old in ("scenarios/s.yaml", "decks/SR.json", "policy.yaml", "classes.yaml"):
        _write(tmp_path, old)

    assert [move.old for move in find_retired(tmp_path).moves] == [
        "classes.yaml",
        "policy.yaml",
        "scenarios",
        "decks",
    ]


def test_a_retired_directory_merges_file_by_file_into_an_existing_successor(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "scenarios/b.yaml")
    _write(tmp_path, "scenarios/a/plot.png")
    _write(tmp_path, "simulation/scenarios/c.yaml")

    found = find_retired(tmp_path)

    assert found.moves == (
        Move(
            root=tmp_path,
            old="scenarios/a/plot.png",
            new="simulation/scenarios/a/plot.png",
            is_dir=False,
            merge=True,
        ),
        Move(
            root=tmp_path,
            old="scenarios/b.yaml",
            new="simulation/scenarios/b.yaml",
            is_dir=False,
            merge=True,
        ),
    )
    assert found.clashes == ()


def test_a_same_named_file_at_both_locations_is_a_clash(tmp_path: Path) -> None:
    _write(tmp_path, "measurement/SR.yaml")
    _write(tmp_path, "measurement/BTS.yaml")
    _write(tmp_path, "simulation/measurement/SR.yaml")

    found = find_retired(tmp_path)

    assert found.clashes == (
        Clash(
            old=tmp_path / "measurement/SR.yaml",
            new=tmp_path / "simulation/measurement/SR.yaml",
        ),
    )
    assert [move.old for move in found.moves] == ["measurement/BTS.yaml"]
    assert not found.empty


def test_an_old_and_a_new_file_both_present_is_a_clash(tmp_path: Path) -> None:
    _write(tmp_path, "classes.yaml")
    _write(tmp_path, "vocabulary/classes.yaml")

    found = find_retired(tmp_path)

    assert found.clashes == (
        Clash(old=tmp_path / "classes.yaml", new=tmp_path / "vocabulary/classes.yaml"),
    )
    assert found.moves == ()


def test_an_old_and_a_new_models_file_both_present_is_a_clash(tmp_path: Path) -> None:
    _write(tmp_path, "models.yaml", "- name: SR\n  engine: pyat\n")
    _write(tmp_path, "simulation/models.yaml", "- name: SR\n  engine: pyat\n")

    found = find_retired(tmp_path)

    assert found.clashes == (
        Clash(old=tmp_path / "models.yaml", new=tmp_path / "simulation/models.yaml"),
    )


def test_an_empty_leftover_directory_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "benchmarks/cross_paradigm").mkdir(parents=True)
    (tmp_path / "decks/sub").mkdir(parents=True)
    _write(tmp_path, "scenarios/.gitkeep")
    _write(tmp_path, "measurement/.cache/SR.yaml")
    _write(tmp_path, ".classes.yaml")

    assert find_retired(tmp_path).empty


def test_hidden_files_are_not_merged(tmp_path: Path) -> None:
    _write(tmp_path, "scenarios/.DS_Store")
    _write(tmp_path, "scenarios/a.yaml")
    _write(tmp_path, "simulation/scenarios/b.yaml")

    assert [move.old for move in find_retired(tmp_path).moves] == ["scenarios/a.yaml"]


def test_deck_values_under_the_retired_prefix_are_edited_at_the_new_models_path(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "models.yaml",
        "# header comment\n"
        "- name: BTS\n"
        "  engine: pyat\n"
        "  deck: decks/BTS.json\n"
        "- deck: 'decks/LTB.json'\n"
        "  name: LTB\n"
        '- {name: SR, deck: "decks/SR.json"}\n'
        "- name: MML\n"
        "  deck: imported/mml/decks/ring.json\n"
        "- name: NODECK\n"
        "  engine: pyat\n"
        "  settings:\n"
        "    deck: decks/not-top-level.json\n",
    )

    found = find_retired(tmp_path)

    new_file = tmp_path / "simulation/models.yaml"
    assert found.deck_edits == (
        DeckEdit(file=new_file, line=4, old="decks/BTS.json", new="simulation/decks/BTS.json"),
        DeckEdit(file=new_file, line=5, old="decks/LTB.json", new="simulation/decks/LTB.json"),
        DeckEdit(file=new_file, line=7, old="decks/SR.json", new="simulation/decks/SR.json"),
    )
    assert found.moves == (
        Move(
            root=tmp_path,
            old="models.yaml",
            new="simulation/models.yaml",
            is_dir=False,
            merge=False,
        ),
    )


def test_deck_values_are_found_in_an_already_moved_models_file(tmp_path: Path) -> None:
    _write(tmp_path, "simulation/models.yaml", "- name: SR\n  deck: decks/SR.json\n")

    found = find_retired(tmp_path)

    assert found.deck_edits == (
        DeckEdit(
            file=tmp_path / "simulation/models.yaml",
            line=2,
            old="decks/SR.json",
            new="simulation/decks/SR.json",
        ),
    )
    assert found.moves == ()
    assert not found.empty


@pytest.mark.parametrize(
    "text",
    ["- name: [unclosed\n", "name: SR\n", "", "- just a string\n", "- deck: 3\n"],
    ids=["invalid", "mapping", "empty", "scalar-entry", "non-string-deck"],
)
def test_a_models_file_without_deck_entries_yields_no_edits(tmp_path: Path, text: str) -> None:
    _write(tmp_path, "simulation/models.yaml", text)

    assert find_retired(tmp_path).empty


def test_imported_layers_are_never_examined(tmp_path: Path) -> None:
    _write(tmp_path, "imported/mml/models.yaml", "- name: SR\n  deck: decks/SR.json\n")
    _write(tmp_path, "imported/mml/decks/SR.json")
    _write(tmp_path, "imported/mml/scenarios/a.yaml")

    assert find_retired(tmp_path).empty


def test_content_retired_paths_are_searched_in_the_content_root_when_mounted(
    tmp_path: Path,
) -> None:
    machine = tmp_path / "machine"
    content = tmp_path / "content"
    _write(machine, "benchmarks/cross_paradigm/queries/tree_queries.json")
    _write(content, "benchmarks/cross_paradigm/queries/tree_queries.json")
    _write(content, "classes.yaml")

    found = find_retired(machine, content)

    assert found.moves == (
        Move(
            root=content,
            old="benchmarks/cross_paradigm/queries",
            new="benchmarks/channel_finder",
            is_dir=True,
            merge=False,
        ),
    )


def test_machine_retired_paths_stay_in_the_machine_root_when_mounted(tmp_path: Path) -> None:
    machine = tmp_path / "machine"
    content = tmp_path / "content"
    _write(machine, "decks/SR.json")
    content.mkdir()

    found = find_retired(machine, content)

    assert found.moves == (
        Move(root=machine, old="decks", new="simulation/decks", is_dir=True, merge=False),
    )


def test_the_result_is_frozen(tmp_path: Path) -> None:
    found = find_retired(tmp_path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        found.moves = ()  # type: ignore[misc]
