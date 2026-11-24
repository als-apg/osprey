"""``navigation.json`` as CANOPY itself reads it, held against OSPREY's check of it.

OSPREY's check (``osprey.integrations.canopy.library``) restates CANOPY's rule
with the standard library, because it runs where CANOPY is not installed. These
cases run against the real ``canopy_mcp`` and hold the two together: the file's
name, the limits, the verdict on a document's shape, and that the check an
image build runs passes a library holding the file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from canopy_mcp import navigation as canopy_navigation

from osprey.cli.templates.manager import TemplateManager
from osprey.integrations.canopy import LIBRARY_SUBPATH, library

HELD = [{"screen": "library/facility/overview", "label": "Start here"}]


def _nested(levels: int) -> list[Any]:
    entries: list[Any] = [{"screen": "library/a"}]
    for _ in range(levels - 1):
        entries = [{"label": "Group", "children": entries}]
    return entries


DOCUMENTS: dict[str, Any] = {
    "empty": [],
    "leaf": [{"screen": "library/a"}],
    "labelled-leaf": [{"screen": "library/a", "label": "A"}],
    "group": [{"label": "Group", "children": [{"screen": "library/a"}]}],
    "group-with-screen": [{"label": "Group", "screen": "library/a", "children": []}],
    "deepest": _nested(library.MAX_NAVIGATION_DEPTH),
    "most-entries": [{"screen": "library/a"}] * library.MAX_NAVIGATION_ENTRIES,
    "longest-label": [{"screen": "library/a", "label": "x" * library.MAX_NAVIGATION_LABEL}],
    "object": {},
    "text": "library/a",
    "entry-not-object": ["library/a"],
    "unknown-key": [{"screen": "library/a", "icon": "x"}],
    "screen-not-text": [{"screen": 7}],
    "screen-blank": [{"screen": " "}],
    "label-empty": [{"screen": "library/a", "label": ""}],
    "label-too-long": [{"screen": "library/a", "label": "x" * (library.MAX_NAVIGATION_LABEL + 1)}],
    "screen-too-long": [{"screen": "library/" + "x" * library.MAX_NAVIGATION_SCREEN}],
    "group-without-label": [{"children": []}],
    "neither": [{"label": "Group"}],
    "children-not-list": [{"label": "Group", "children": {}}],
    "too-deep": _nested(library.MAX_NAVIGATION_DEPTH + 1),
    "too-many": [{"screen": "library/a"}] * (library.MAX_NAVIGATION_ENTRIES + 1),
}


def _canopy_fault(document: Any) -> list[str]:
    try:
        canopy_navigation.check_navigation(document)
    except canopy_navigation.InvalidNavigation as fault:
        return [str(fault)]
    return []


def test_the_file_has_the_name_canopy_reads() -> None:
    assert library.NAVIGATION_NAME == canopy_navigation.NAVIGATION_FILE


@pytest.mark.parametrize("limit", ["BYTES", "DEPTH", "ENTRIES", "LABEL", "SCREEN"])
def test_a_limit_is_the_one_canopy_states(limit: str) -> None:
    name = f"MAX_NAVIGATION_{limit}"
    assert getattr(library, name) == getattr(canopy_navigation, name)


@pytest.mark.parametrize("name", sorted(DOCUMENTS))
def test_the_shape_is_judged_as_canopy_judges_it(name: str) -> None:
    document = DOCUMENTS[name]
    assert library.navigation_faults(document, ["a"]) == _canopy_fault(document)


def test_canopy_hands_on_a_file_this_check_passes(tmp_path: Path) -> None:
    (tmp_path / library.NAVIGATION_NAME).write_text(json.dumps(HELD), "utf-8")
    assert library.navigation_faults(HELD, ["facility/overview"]) == []
    assert canopy_navigation.NavigationFile(tmp_path).read() == HELD


def test_the_image_build_check_passes_a_library_holding_the_file(tmp_path: Path) -> None:
    """The shipped library with a ``navigation.json`` beside ``screens/``.

    ``osprey build`` passes it, and so must the check the image build runs on
    the tree it copied in; otherwise the file could not reach a served library.
    """
    shipped = Path(TemplateManager().template_root) / "apps" / "demo" / LIBRARY_SUBPATH
    root = tmp_path / "library"
    shutil.copytree(shipped, root)
    navigation = root / library.NAVIGATION_NAME
    navigation.write_text(json.dumps(HELD, indent=2) + "\n", "utf-8")
    navigation.chmod(0o644)
    assert library.library_is_fresh(root)
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "canopy_mcp.report",
            str(root),
            "--served-at",
            "library",
            "--check",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
