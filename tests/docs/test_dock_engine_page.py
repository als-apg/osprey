"""The dock-engine contract page against what OSPREY actually holds.

``reference/contracts/dock-engine.rst`` names the vendored engine files, their
manifest entries, the server's layout rules and the copied cases. Each is a path
or a name in the tree, so this module checks that every one the page names
exists, and that the manifest entries are the committed, never-fetched kind the
page describes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from osprey.interfaces import vendor
from osprey.interfaces.web_terminal import dock_layout

_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _ROOT / "docs/source/reference/contracts/dock-engine.rst"
_VENDOR_DIR = "src/osprey/interfaces/web_terminal/static/vendor/"
_MANIFEST = "src/osprey/interfaces/vendor_manifest.json"
_ENGINE_FILES = ("dock-engine.mjs", "dock-engine.d.mts")
_RULES = ("validate_layout", "layout_refusal", "normalize_layout", "LAYOUT_REASONS")
_DOCK_TESTS = ("cases-pin.json", "test_cases_pin.py", "test_case_replay.py")


def _page() -> str:
    return " ".join(_PAGE.read_text(encoding="utf-8").split())


def _literals() -> set[str]:
    return set(re.findall(r"``([^`]+)``", _page()))


def test_the_page_names_where_the_files_live() -> None:
    names = _literals()
    assert {_VENDOR_DIR, _MANIFEST, "tests/dock_engine/cases/"} <= names
    assert Path(vendor.MANIFEST_PATH).resolve() == (_ROOT / _MANIFEST).resolve()


@pytest.mark.parametrize("path", [_VENDOR_DIR, _MANIFEST, "tests/dock_engine/cases/"])
def test_each_named_path_exists(path: str) -> None:
    assert (_ROOT / path).exists(), path


@pytest.mark.parametrize("filename", _ENGINE_FILES)
def test_each_engine_file_is_vendored_and_named(filename: str) -> None:
    assert filename in _literals()
    assert (_ROOT / _VENDOR_DIR / filename).is_file()


@pytest.mark.parametrize("filename", _ENGINE_FILES)
def test_each_engine_file_has_a_committed_manifest_entry(filename: str) -> None:
    """A source commit and no URL: the kind ``osprey vendor fetch`` refuses to fetch."""
    assets = json.loads((_ROOT / _MANIFEST).read_text(encoding="utf-8"))["assets"]
    (asset,) = [a for a in assets if a["filename"] == filename]
    assert "url" not in asset
    assert re.fullmatch(r"[0-9a-f]{40}", asset["source"]["commit"])
    assert "re-copy" in vendor._committed_asset_problem(asset, _VENDOR_DIR).lower()


@pytest.mark.parametrize("name", _RULES)
def test_each_layout_rule_the_page_names_exists(name: str) -> None:
    assert name in _literals()
    assert hasattr(dock_layout, name)


def test_a_planted_rule_is_caught() -> None:
    assert not hasattr(dock_layout, "planted_layout_rule")


@pytest.mark.parametrize("name", _DOCK_TESTS)
def test_each_dock_engine_test_file_the_page_names_exists(name: str) -> None:
    assert name in _literals()
    assert (_ROOT / "tests/dock_engine" / name).is_file()


def test_the_case_directories_are_the_two_the_page_counts() -> None:
    cases = _ROOT / "tests/dock_engine/cases"
    assert sorted(p.name for p in cases.iterdir() if p.is_dir()) == ["normalize", "validate"]
    assert "the two case directories" in _page()
