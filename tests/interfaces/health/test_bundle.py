"""Static validation of the shipped System Health dashboard bundle.

The web terminal mounts ``src/osprey/interfaces/health/static`` as the
``system-health`` panel, so it must satisfy the same fail-closed panel
contract every other shipped surface does: a schema-valid ``manifest.json``,
an entry HTML that opts into shared design-system theming, and zero raw hex
color literals anywhere in its HTML/CSS/JS (it themes only through
``var(--…)`` tokens).

These mirror the reference-panel integration checks in
``tests/interfaces/design_system/test_panel_validator.py`` against the real
bundle directory rather than a synthetic one — if the bundle ever drifts
(a stray hex literal, a dropped design-system link, a malformed manifest),
the validator catches it here before it can ship.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import osprey
from osprey.interfaces.design_system.panels.validator import (
    MANIFEST_FILENAME,
    PanelRule,
    PanelValidationError,
    assert_valid_panel,
    validate_panel,
)

# The bundle ships inside the osprey package; locate it from the package
# root (a filesystem path — no import of the interfaces.health subpackage,
# which the later app tasks add).
_BUNDLE_DIR = Path(osprey.__file__).parent / "interfaces" / "health" / "static"


def test_bundle_directory_is_present() -> None:
    # A missing bundle would make the checks below vacuously pass; a
    # standalone presence assertion keeps that failure loud and specific.
    assert (_BUNDLE_DIR / MANIFEST_FILENAME).is_file()
    assert (_BUNDLE_DIR / "index.html").is_file()
    assert (_BUNDLE_DIR / "dashboard.css").is_file()


def test_bundle_validates_with_no_errors() -> None:
    assert validate_panel(_BUNDLE_DIR) == []


def test_assert_valid_panel_does_not_raise_for_bundle() -> None:
    try:
        assert_valid_panel(_BUNDLE_DIR)  # no exception
    except PanelValidationError as exc:  # pragma: no cover - failure detail
        pytest.fail(f"bundle failed panel validation:\n{exc}")


def test_bundle_entry_links_design_system_and_fonts() -> None:
    # Guard the references a reviewer relies on the entry HTML to carry: the
    # pre-paint boot script, the token stylesheet, and the shared web font.
    entry_html = (_BUNDLE_DIR / "index.html").read_text(encoding="utf-8")
    assert "/design-system/js/theme-boot.js" in entry_html
    assert "/design-system/css/tokens.css" in entry_html
    assert "/static/fonts/fonts.css" in entry_html


def test_bundle_is_token_only() -> None:
    # The reason this panel exists as a token port: no raw hex color may
    # survive in the HTML or the sibling CSS.
    hex_errors = [
        error for error in validate_panel(_BUNDLE_DIR) if error.rule is PanelRule.RAW_HEX_COLOR
    ]
    assert hex_errors == []
