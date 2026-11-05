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
import sys
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


def _rule_names() -> set[str]:
    """The names the page says ``dock_layout`` holds: the literals of the rules row."""
    row = re.search(r"\*\*The server's layout rules\*\*:(.*?) - The engine's cases", _page())
    assert row is not None, "the layout-rules row is missing"
    return set(re.findall(r"``([A-Za-z_]+)``", row.group(1)))


def _repo_paths() -> set[str]:
    """Every literal on the page that names a path in the repository."""
    return {name for name in _literals() if name.startswith(("src/", "tests/"))}


def test_the_page_names_where_the_files_live() -> None:
    names = _literals()
    assert {_VENDOR_DIR, _MANIFEST, "tests/dock_engine/cases/"} <= names
    assert Path(vendor.MANIFEST_PATH).resolve() == (_ROOT / _MANIFEST).resolve()


def test_each_named_path_exists() -> None:
    missing = sorted(p for p in _repo_paths() if not (_ROOT / p).exists())
    assert not missing, f"dock-engine.rst names paths that do not exist: {missing}"


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


def test_the_layout_rules_the_page_names_exist() -> None:
    assert set(_RULES) <= _rule_names()
    missing = sorted(n for n in _rule_names() - {"dock_layout"} if not hasattr(dock_layout, n))
    assert not missing, f"dock-engine.rst names dock_layout members that do not exist: {missing}"


@pytest.fixture
def planted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the page readers at a copy of the page with *old* replaced by *new*."""

    def plant(old: str, new: str) -> None:
        text = _PAGE.read_text(encoding="utf-8")
        assert text.count(old) == 1, old
        copy = tmp_path / "dock-engine.rst"
        copy.write_text(text.replace(old, new), encoding="utf-8")
        monkeypatch.setattr(sys.modules[__name__], "_PAGE", copy)

    return plant


def test_a_planted_rule_is_reported(planted) -> None:
    planted("``normalize_layout``", "``normalize_layout``, ``planted_layout_rule``")
    with pytest.raises(AssertionError, match="planted_layout_rule"):
        test_the_layout_rules_the_page_names_exist()


def test_a_planted_path_is_reported(planted) -> None:
    planted("``tests/dock_engine/cases/``", "``tests/dock_engine/planted/``")
    with pytest.raises(AssertionError, match="planted"):
        test_each_named_path_exists()


@pytest.mark.parametrize("name", _DOCK_TESTS)
def test_each_dock_engine_test_file_the_page_names_exists(name: str) -> None:
    assert name in _literals()
    assert (_ROOT / "tests/dock_engine" / name).is_file()


def test_the_case_directories_are_the_two_the_page_counts() -> None:
    cases = _ROOT / "tests/dock_engine/cases"
    assert sorted(p.name for p in cases.iterdir() if p.is_dir()) == ["normalize", "validate"]
    assert "the two case directories" in _page()
