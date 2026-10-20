"""The trajectory panel bundle is a valid design-system panel.

The bundle ships a ``manifest.json``, links the shared theme assets and styles
through tokens only. Its ``vendor/`` directory holds third-party builds fetched
at image build time (Plotly), which are not panel source and carry their own
colour literals, so the validator's token-only scan skips that segment and
nothing else.
"""

from __future__ import annotations

import json
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
