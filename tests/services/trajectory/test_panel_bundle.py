"""The trajectory panel bundle is a valid design-system panel.

The bundle ships a ``manifest.json``, links the shared theme assets and styles
through tokens only. Its ``vendor/`` directory holds third-party builds fetched
at image build time (Plotly), which are not panel source and carry their own
colour literals, so the validator's token-only scan skips that segment and
nothing else.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from osprey.interfaces.design_system.panels.validator import (
    PanelRule,
    assert_valid_panel,
    validate_panel,
)

PANEL_DIR = Path(__file__).resolve().parents[3] / "src/osprey/services/trajectory/panel"


def test_the_trajectory_bundle_is_a_valid_panel() -> None:
    assert_valid_panel(PANEL_DIR)


def test_the_manifest_names_the_trajectory_panel_and_its_entry() -> None:
    manifest = json.loads((PANEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "trajectory"
    assert manifest["entry"] == "index.html"
    assert (PANEL_DIR / manifest["entry"]).is_file()


@pytest.fixture
def bundle_copy(tmp_path: Path) -> Path:
    """A copy of the bundle without any fetched vendor files."""
    copy = tmp_path / "panel"
    shutil.copytree(PANEL_DIR, copy, ignore=shutil.ignore_patterns("vendor", "node_modules"))
    return copy


def _hex_findings(panel_dir: Path) -> list[str]:
    return [f.source for f in validate_panel(panel_dir) if f.rule is PanelRule.RAW_HEX_COLOR]


def test_a_vendored_build_with_colour_literals_does_not_fail_the_panel(bundle_copy: Path) -> None:
    vendor = bundle_copy / "vendor"
    vendor.mkdir()
    (vendor / "plotly-3.3.1.min.js").write_text('var c="#1f77b4",d="#fff";\n', encoding="utf-8")
    (vendor / "nested").mkdir()
    (vendor / "nested" / "theme.css").write_text("a{color:#abcdef}\n", encoding="utf-8")

    assert_valid_panel(bundle_copy)


def test_the_scan_still_covers_panel_source_beside_the_vendor_dir(bundle_copy: Path) -> None:
    """Only a ``vendor`` path segment is skipped: a file merely named like one is panel source."""
    (bundle_copy / "vendor").mkdir()
    (bundle_copy / "vendored.js").write_text('const c = "#123456";\n', encoding="utf-8")
    (bundle_copy / "colours.css").write_text("a { color: #abc; }\n", encoding="utf-8")

    sources = _hex_findings(bundle_copy)

    assert any(source.endswith("vendored.js:1") for source in sources), sources
    assert any(source.endswith("colours.css:1") for source in sources), sources


def test_a_vendor_segment_above_the_panel_dir_does_not_hide_the_panel(tmp_path: Path) -> None:
    """The skip is relative to the panel dir; a panel living under a ``vendor`` path is scanned."""
    panel = tmp_path / "vendor" / "panel"
    shutil.copytree(PANEL_DIR, panel, ignore=shutil.ignore_patterns("vendor", "node_modules"))
    (panel / "colours.css").write_text("a { color: #abc; }\n", encoding="utf-8")

    assert any(source.endswith("colours.css:1") for source in _hex_findings(panel))


#: A facility's own vocabulary: its name, its ring's address prefix and the
#: words it uses for places along the ring. Every label, place and level the
#: panel shows comes from the facility model through ``Snapshot.catalog``, so
#: none of them may be spelled in the panel's own source.
FACILITY_WORDS = re.compile(
    r"(?i)\b(?:sector|sect\d*|girder|straight)s?\b|\bSR[:/]|\bALS(?:-U)?\b"
)


def _panel_sources() -> list[Path]:
    return sorted(
        path
        for path in PANEL_DIR.rglob("*")
        if path.suffix in {".js", ".css", ".html"}
        and "vendor" not in path.relative_to(PANEL_DIR).parts
        and not path.name.endswith(".test.js")
    )


def test_there_is_panel_source_to_scan() -> None:
    names = {path.name for path in _panel_sources()}
    assert {"setup.js", "figures.js", "panel.css"} <= names


@pytest.mark.parametrize("path", _panel_sources(), ids=lambda path: path.name)
def test_the_panel_source_names_no_facility_word(path: Path) -> None:
    hits = [
        f"{path.name}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FACILITY_WORDS.search(line)
    ]
    assert hits == []


@pytest.mark.parametrize("text", ["sector 3", "SECT12", "SR:BPM:1", "SR/SECT1", "ALS", "girders"])
def test_the_facility_word_scan_catches_a_facility_name(text: str) -> None:
    assert FACILITY_WORDS.search(text)


@pytest.mark.parametrize("text", ["select", "selection", "insertion", "cell(value)", "SRC", "falsey"])
def test_the_facility_word_scan_leaves_generic_words_alone(text: str) -> None:
    assert not FACILITY_WORDS.search(text)
