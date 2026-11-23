"""A library's ``navigation.json``: admitted at the root as a regular file, and checked.

The file is optional. The listing admits it beside ``screens/`` and the report
files and keeps it off the list; the freshness check, which ``osprey build``
runs before it builds anything, reads it and refuses a navigation the viewer
would not draw as written. None of this needs CANOPY installed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from osprey.integrations.canopy import library, stage
from osprey.integrations.canopy.library import NAVIGATION_NAME

SHA = "0" * 64
VALID = [
    {"screen": "library/overview", "label": "Start here"},
    {
        "label": "Magnets",
        "screen": "library/magnets/all",
        "children": [
            {"screen": "library/magnets/sector-1", "label": "Sector 1"},
            {"label": "Spares", "children": [{"screen": "library/magnets/spare"}]},
        ],
    },
]
SCREENS = ("overview", "magnets/all", "magnets/sector-1", "magnets/spare")
ADVICE = "correct navigation.json, or delete it and the sidebar shows the folder tree"


def _entry(path: str, kind: str = "file") -> library.Entry:
    entry: library.Entry = {"path": path, "kind": kind, "mode": 0o644}
    if kind == "file":
        entry["sha256"] = SHA
    return entry


def _report(root: Path) -> None:
    judged = library.judge(library.scan(root))
    report = {
        "generator": {"schema": "canopy-library-report/1", "canopy_version": "0.0.0"},
        "files": {
            "digest": judged.digest,
            "list": [{"path": path, "sha256": sha} for path, sha in judged.files],
        },
    }
    (root / library.REPORT_JSON).write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    (root / library.REPORT_MD).write_bytes(b"# report\n")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A library holding the screens :data:`VALID` names, with a current report."""
    root = tmp_path / "library"
    for screen in SCREENS:
        path = root / "screens" / f"{screen}.screen.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"kind": "screen"}\n')
    _report(root)
    for path in (root, *root.rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644)
    return root


def _navigation(root: Path, content: Any) -> Path:
    path = root / NAVIGATION_NAME
    data = content if isinstance(content, bytes) else json.dumps(content).encode("utf-8")
    path.write_bytes(data)
    path.chmod(0o644)
    return path


def _refused(root: Path) -> list[str]:
    verdict = library.library_is_fresh(root)
    assert not verdict
    assert all(line.isascii() and "\n" not in line for line in verdict.lines)
    return list(verdict.lines)


# --- the listing ----------------------------------------------------------------------


def test_the_name_is_fixed() -> None:
    assert NAVIGATION_NAME == "navigation.json"
    assert NAVIGATION_NAME not in library.REPORT_NAMES


def test_the_listing_admits_the_file_and_keeps_it_off_the_list() -> None:
    screen = _entry("screens/overview.screen.json")
    without = library.judge([_entry("screens", "dir"), screen])
    judged = library.judge([_entry("screens", "dir"), screen, _entry(NAVIGATION_NAME)])
    assert judged.refusals == ()
    assert judged.files == without.files and judged.digest == without.digest


def test_a_symlink_of_that_name_is_refused_as_a_symlink() -> None:
    judged = library.judge([_entry("screens", "dir"), _entry(NAVIGATION_NAME, "symlink")])
    assert judged.refusals == ({"path": NAVIGATION_NAME, "code": "symlink"},)


@pytest.mark.parametrize("kind", ["dir", "other"])
def test_anything_else_of_that_name_is_outside_screens(kind: str) -> None:
    judged = library.judge([_entry("screens", "dir"), _entry(NAVIGATION_NAME, kind)])
    assert judged.refusals == ({"path": NAVIGATION_NAME, "code": "outside_screens"},)


@pytest.mark.parametrize(
    "name", ["Navigation.json", "navigation.JSON", "navigation.yaml", "navigation", "nav.json"]
)
def test_any_other_root_file_is_still_refused(name: str) -> None:
    judged = library.judge([_entry("screens", "dir"), _entry(name)])
    assert judged.refusals == ({"path": name, "code": "outside_screens"},)


def test_the_name_means_nothing_below_screens() -> None:
    judged = library.judge([_entry("screens", "dir"), _entry(f"screens/{NAVIGATION_NAME}")])
    assert judged.refusals == ({"path": f"screens/{NAVIGATION_NAME}", "code": "suffix"},)


def test_the_refusal_of_a_root_entry_names_what_a_root_holds() -> None:
    text = library.refusal_text({"path": "notes.txt", "code": "outside_screens"})
    assert "screens/, the LIBRARY-REPORT files and navigation.json" in text


def test_the_image_build_context_keeps_the_name() -> None:
    assert not library.is_ignored_name(NAVIGATION_NAME)


# --- the freshness check ----------------------------------------------------------------


def test_a_library_without_the_file_is_fresh(root: Path) -> None:
    assert library.library_is_fresh(root).lines == ("report is current (4 files)",)


def test_a_library_with_a_valid_file_is_fresh(root: Path) -> None:
    _navigation(root, VALID)
    assert library.library_is_fresh(root).lines == ("report is current (4 files)",)


def test_the_file_does_not_make_the_report_stale(root: Path) -> None:
    before = library.library_list(root)
    _navigation(root, VALID)
    assert library.library_list(root) == before
    _navigation(root, [])
    assert library.library_is_fresh(root)


def test_the_build_step_passes_a_library_with_a_valid_file(tmp_path: Path, root: Path) -> None:
    _navigation(root, VALID)
    deployment = tmp_path / "deployment"
    (deployment / "data" / "canopy").mkdir(parents=True)
    root.rename(deployment / "data" / "canopy" / "library")
    stage.check_library(deployment)


def test_the_build_step_refuses_a_library_with_an_invalid_file(tmp_path: Path, root: Path) -> None:
    _navigation(root, {"screen": "library/overview"})
    deployment = tmp_path / "deployment"
    (deployment / "data" / "canopy").mkdir(parents=True)
    root.rename(deployment / "data" / "canopy" / "library")
    with pytest.raises(stage.StaleLibrary) as refused:
        stage.check_library(deployment)
    assert "navigation: navigation.json: navigation: expected a list of entries" in str(
        refused.value
    )


def test_a_symlinked_file_is_refused_and_not_read(root: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("not json", "utf-8")
    os.symlink(elsewhere, root / NAVIGATION_NAME)
    assert _refused(root) == [
        "symlink: navigation.json: a symlink; delete it, or replace it with the file it points at"
    ]


def test_a_directory_of_that_name_is_refused(root: Path) -> None:
    (root / NAVIGATION_NAME).mkdir(mode=0o755)
    lines = _refused(root)
    assert len(lines) == 1 and lines[0].startswith("outside_screens: navigation.json: ")


@pytest.mark.parametrize(
    ("content", "fault"),
    [
        (b"[", "not JSON ("),
        (b"\xff\xfe", "not JSON ("),
        (b"[" * 100_000, "not JSON ("),
        (b"{}", "navigation: expected a list of entries"),
        (b'"library/overview"', "navigation: expected a list of entries"),
        (b"[" + b" " * library.MAX_NAVIGATION_BYTES + b"]", "larger than the 262144-byte limit"),
    ],
    ids=["truncated", "not-utf8", "too-deep-to-parse", "object", "text", "too-large"],
)
def test_a_file_that_is_no_navigation_is_refused(root: Path, content: bytes, fault: str) -> None:
    _navigation(root, content)
    lines = _refused(root)
    assert lines[0].startswith(f"navigation: navigation.json: {fault}")
    assert lines[1:] == [ADVICE]


def test_a_dangling_screen_is_refused_by_where_it_is(root: Path) -> None:
    (root / "screens" / "magnets" / "spare.screen.json").unlink()
    _report(root)
    _navigation(root, VALID)
    assert _refused(root) == [
        "navigation: navigation.json: navigation[1].children[1].children[0].screen: "
        "the library holds no screen 'library/magnets/spare'",
        ADVICE,
    ]


def test_every_dangling_screen_is_named(root: Path) -> None:
    _navigation(
        root,
        [
            {"screen": "library/absent"},
            {"screen": "library/overview"},
            {"label": "Group", "screen": "library/also-absent", "children": []},
        ],
    )
    lines = _refused(root)
    assert [line.split(": ")[2] for line in lines[:-1]] == [
        "navigation[0].screen",
        "navigation[2].screen",
    ]


def test_a_screen_the_listing_refuses_is_not_held(root: Path) -> None:
    """A screen under a refused name is off the list, so an entry naming it dangles."""
    hidden = root / "screens" / ".hidden.screen.json"
    hidden.write_bytes(b"{}\n")
    hidden.chmod(0o644)
    _navigation(root, [{"screen": "library/.hidden"}])
    lines = _refused(root)
    assert "dotfile: screens/.hidden.screen.json: a dotfile; delete it" in lines
    assert any("the library holds no screen 'library/.hidden'" in line for line in lines)


def test_a_fault_with_a_name_outside_ascii_is_one_ascii_line(root: Path) -> None:
    _navigation(root, [{"screen": "library/über\nsicht"}])
    lines = _refused(root)
    assert "library/\\xfcber\\\\nsicht" in lines[0]


def test_the_file_is_held_to_the_modes_the_image_needs(root: Path) -> None:
    _navigation(root, VALID).chmod(0o600)
    lines = _refused(root)
    assert lines[0] == "mode: navigation.json: not readable by others (mode 0600)"


def test_a_navigation_fault_comes_with_every_other_refusal(root: Path) -> None:
    _navigation(root, [{"screen": "library/absent"}])
    (root / "README.md").write_bytes(b"r")
    (root / "README.md").chmod(0o644)
    lines = _refused(root)
    assert lines[0].startswith("outside_screens: README.md: ")
    assert lines[1].startswith("navigation: navigation.json: navigation[0].screen: ")
    assert lines[2] == ADVICE


# --- the shape and the limits -------------------------------------------------------------


def test_a_valid_navigation_has_no_fault() -> None:
    assert library.navigation_faults(VALID, SCREENS) == []
    assert library.navigation_faults([], ()) == []


def _nested(levels: int) -> list[Any]:
    """A navigation whose innermost list of entries sits *levels* deep."""
    entries: list[Any] = [{"screen": "library/overview"}]
    for _ in range(levels - 1):
        entries = [{"label": "Group", "children": entries}]
    return entries


@pytest.mark.parametrize(
    ("document", "fault"),
    [
        (["library/overview"], "navigation[0]: expected an object"),
        ([{"screen": "library/overview", "icon": "x"}], "navigation[0]: unknown key 'icon'"),
        ([{"screen": 7}], "navigation[0].screen: expected non-empty text"),
        ([{"screen": "  "}], "navigation[0].screen: expected non-empty text"),
        (
            [{"screen": "library/overview", "label": ""}],
            "navigation[0].label: expected non-empty text",
        ),
        (
            [{"screen": "library/overview", "label": "x" * 121}],
            "navigation[0].label: longer than 120 characters",
        ),
        (
            [{"screen": "library/" + "x" * 505}],
            "navigation[0].screen: longer than 512 characters",
        ),
        ([{"label": "Group"}], "navigation[0]: needs a screen or children"),
        ([{"children": []}], "navigation[0]: a group needs a label"),
        (
            [{"label": "Group", "children": {}}],
            "navigation[0].children: expected a list of entries",
        ),
        (
            [{"label": "Group", "children": [{}]}],
            "navigation[0].children[0]: needs a screen or children",
        ),
        (
            _nested(9),
            "navigation" + "[0].children" * 8 + ": nested deeper than 8 levels",
        ),
        ([{"screen": "library/overview"}] * 2001, "more than 2000 entries"),
    ],
)
def test_the_first_fault_of_the_shape_is_the_answer(document: Any, fault: str) -> None:
    assert library.navigation_faults(document, SCREENS) == [fault]


def test_the_limits_themselves_are_within_bounds() -> None:
    assert library.navigation_faults(_nested(8), SCREENS) == []
    assert library.navigation_faults([{"screen": "library/overview"}] * 2000, SCREENS) == []
    assert (
        library.navigation_faults([{"screen": "library/overview", "label": "x" * 120}], SCREENS)
        == []
    )


@pytest.mark.parametrize(
    "screen", ["overview", "drafts/overview", "/library/overview", "Library/overview"]
)
def test_a_screen_named_outside_the_library_is_a_fault(screen: str) -> None:
    assert library.navigation_faults([{"screen": screen}], SCREENS) == [
        f"navigation[0].screen: {screen!r} is not a screen of the library; write it as library/<id>"
    ]


@pytest.mark.parametrize("screen", ["library/", "library/Overview", "library/overview.screen.json"])
def test_a_screen_the_library_does_not_hold_is_a_fault(screen: str) -> None:
    assert library.navigation_faults([{"screen": screen}], SCREENS) == [
        f"navigation[0].screen: the library holds no screen {screen!r}"
    ]
