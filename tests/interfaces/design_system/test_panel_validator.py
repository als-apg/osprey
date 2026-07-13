"""Tests for osprey.interfaces.design_system.panels.validator.

Builds small, synthetic panel directories under ``tmp_path`` (no
dependency on any real reference panel — that is a separate task, which
adds its own integration tests to this file afterward) to exercise each
fail-closed static check in isolation:

- a fully-compliant panel validates with zero errors;
- a missing ``manifest.json`` errors;
- a schema-invalid manifest folds the manifest validator's errors in;
- a manifest whose ``entry`` file is absent errors;
- an entry HTML missing the design-system ``<link>`` or the theme-boot
  ``<script>`` errors;
- a raw hex color literal in the HTML/CSS is flagged with its location,
  while a ``var(--…)``-only panel and a hex-shaped *fragment* anchor
  (``href="#section"``) do not false-positive;
- :func:`validate_panel` returns *all* errors for a multiply-broken panel;
- :func:`assert_valid_panel` raises :class:`PanelValidationError` whose
  ``str()`` lists every failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.interfaces.design_system import panels as panels_pkg
from osprey.interfaces.design_system.panels.validator import (
    MANIFEST_FILENAME,
    PanelError,
    PanelRule,
    PanelValidationError,
    assert_valid_panel,
    validate_panel,
)

# The two design-system references a compliant entry HTML head must carry,
# mirroring the reference markup in osprey.interfaces.artifacts.app.
_THEME_BOOT_TAG = '<script src="/design-system/js/theme-boot.js"></script>'
_TOKENS_CSS_TAG = '<link rel="stylesheet" href="/design-system/css/tokens.css">'
_FONTS_CSS_TAG = '<link rel="stylesheet" href="/static/fonts/fonts.css">'


def _compliant_html(*, body: str = "") -> str:
    """A minimal, fully-compliant panel entry HTML document."""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        f"{_THEME_BOOT_TAG}\n"
        f"{_TOKENS_CSS_TAG}\n"
        f"{_FONTS_CSS_TAG}\n"
        "<style>\n"
        "  body { background: var(--bg-primary); color: var(--text-primary); }\n"
        "</style>\n"
        "</head>\n"
        f"<body>{body}</body>\n"
        "</html>\n"
    )


def _write_panel(
    panel_dir: Path,
    *,
    manifest: dict | str | None = None,
    entry_name: str = "index.html",
    entry_html: str | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Write a synthetic panel directory and return it.

    Args:
        panel_dir: Directory to create the panel in.
        manifest: The manifest object to serialize (or a raw string to
            write verbatim, e.g. malformed JSON). ``None`` writes a valid
            manifest pointing at ``entry_name``.
        entry_name: The manifest's declared entry filename.
        entry_html: The entry HTML to write. ``None`` (the default) skips
            writing the entry file entirely (to exercise the missing-entry
            check); callers pass ``_compliant_html()`` for the happy path.
        extra_files: Extra ``{relative_name: text}`` files to write.
    """
    panel_dir.mkdir(parents=True, exist_ok=True)

    if manifest is None:
        manifest = {"id": "demo-panel", "label": "Demo Panel", "entry": entry_name}
    if isinstance(manifest, str):
        (panel_dir / MANIFEST_FILENAME).write_text(manifest, encoding="utf-8")
    else:
        (panel_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

    if entry_html is not None:
        (panel_dir / entry_name).write_text(entry_html, encoding="utf-8")

    for name, text in (extra_files or {}).items():
        (panel_dir / name).write_text(text, encoding="utf-8")

    return panel_dir


def _rules(errors: list[PanelError]) -> set[PanelRule]:
    return {error.rule for error in errors}


# --- Happy path ---------------------------------------------------------------


def test_fully_compliant_panel_has_no_errors(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "panel", entry_html=_compliant_html())
    assert validate_panel(panel) == []


def test_assert_valid_panel_does_not_raise_for_compliant_panel(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "panel", entry_html=_compliant_html())
    assert_valid_panel(panel)  # no exception


def test_compliant_panel_with_token_only_css_sibling_passes(tmp_path: Path) -> None:
    panel = _write_panel(
        tmp_path / "panel",
        entry_html=_compliant_html(),
        extra_files={"panel.css": ".widget { color: var(--accent-base); }\n"},
    )
    assert validate_panel(panel) == []


# --- Check 1: manifest present + schema-valid + entry exists -------------------


def test_missing_manifest_errors(tmp_path: Path) -> None:
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "index.html").write_text(_compliant_html(), encoding="utf-8")

    errors = validate_panel(panel)

    assert _rules(errors) == {PanelRule.MANIFEST_MISSING}
    assert MANIFEST_FILENAME in errors[0].message


def test_malformed_json_manifest_errors(tmp_path: Path) -> None:
    panel = _write_panel(
        tmp_path / "panel", manifest="{ not valid json", entry_html=_compliant_html()
    )

    errors = validate_panel(panel)

    assert PanelRule.MANIFEST_INVALID in _rules(errors)
    assert any("not valid JSON" in error.message for error in errors)


def test_schema_invalid_manifest_folds_manifest_errors(tmp_path: Path) -> None:
    # Missing required 'label' and a bad (non-kebab) id: manifest.py's
    # schema validator produces >1 error, and all of them must be folded
    # in under the single MANIFEST_INVALID rule.
    panel = _write_panel(
        tmp_path / "panel",
        manifest={"id": "Bad Id", "entry": "index.html"},
        entry_html=_compliant_html(),
    )

    errors = validate_panel(panel)

    assert _rules(errors) == {PanelRule.MANIFEST_INVALID}
    assert len(errors) >= 2  # bad id + missing label
    messages = " ".join(error.message for error in errors)
    assert "label" in messages
    assert "id" in messages


def test_missing_entry_file_errors(tmp_path: Path) -> None:
    # Manifest is schema-valid, but the entry file it points at is absent.
    panel = _write_panel(
        tmp_path / "panel",
        manifest={"id": "demo-panel", "label": "Demo", "entry": "index.html"},
        entry_html=None,  # do not write the entry file
    )

    errors = validate_panel(panel)

    assert _rules(errors) == {PanelRule.ENTRY_MISSING}
    assert "index.html" in errors[0].message


# --- Check 2: design-system linked --------------------------------------------


def test_missing_design_system_link_errors(tmp_path: Path) -> None:
    html = _compliant_html().replace(_TOKENS_CSS_TAG, "")
    panel = _write_panel(tmp_path / "panel", entry_html=html)

    errors = validate_panel(panel)

    assert _rules(errors) == {PanelRule.MISSING_DESIGN_SYSTEM_LINK}


def test_missing_theme_boot_script_errors(tmp_path: Path) -> None:
    html = _compliant_html().replace(_THEME_BOOT_TAG, "")
    panel = _write_panel(tmp_path / "panel", entry_html=html)

    errors = validate_panel(panel)

    assert _rules(errors) == {PanelRule.MISSING_THEME_BOOT}


def test_missing_font_link_errors(tmp_path: Path) -> None:
    html = _compliant_html().replace(_FONTS_CSS_TAG, "")
    panel = _write_panel(tmp_path / "panel", entry_html=html)

    errors = validate_panel(panel)

    assert _rules(errors) == {PanelRule.MISSING_FONT_LINK}


def test_design_system_link_matches_regardless_of_attribute_order(tmp_path: Path) -> None:
    # href before rel, single quotes, and src on the boot script: attribute
    # order and quote style must not matter — only the load-bearing path.
    html = _compliant_html()
    html = html.replace(
        _TOKENS_CSS_TAG, "<link href='/design-system/css/tokens.css' rel='stylesheet'>"
    )
    panel = _write_panel(tmp_path / "panel", entry_html=html)

    assert validate_panel(panel) == []


# --- Check 3: token-only (no raw hex colors) ----------------------------------


def test_raw_hex_color_in_html_is_flagged_with_location(tmp_path: Path) -> None:
    html = _compliant_html(body='\n<div style="color: #ff0000;">hi</div>\n')
    panel = _write_panel(tmp_path / "panel", entry_html=html)

    errors = validate_panel(panel)

    assert _rules(errors) == {PanelRule.RAW_HEX_COLOR}
    (error,) = errors
    assert "#ff0000" in error.message
    # source is "path:line"; the literal sits on its own line in the body.
    assert error.source.startswith(str(panel / "index.html"))
    assert error.source.rsplit(":", 1)[1].isdigit()


def test_raw_hex_color_in_css_sibling_is_flagged(tmp_path: Path) -> None:
    panel = _write_panel(
        tmp_path / "panel",
        entry_html=_compliant_html(),
        extra_files={"panel.css": ".x { color: var(--ok); }\n.y { color: #abc; }\n"},
    )

    errors = validate_panel(panel)

    assert _rules(errors) == {PanelRule.RAW_HEX_COLOR}
    (error,) = errors
    assert "#abc" in error.message
    assert error.source.startswith(str(panel / "panel.css"))
    # Second line of the css file.
    assert error.source.endswith(":2")


@pytest.mark.parametrize("hex_literal", ["#fff", "#0f08", "#14b8a6", "#14b8a680"])
def test_all_valid_color_hex_lengths_are_flagged(tmp_path: Path, hex_literal: str) -> None:
    # 3/4/6/8-digit hex are the color shapes validate.py accepts; each must
    # be caught as a raw color.
    panel = _write_panel(
        tmp_path / "panel",
        entry_html=_compliant_html(),
        extra_files={"panel.css": f"a {{ color: {hex_literal}; }}\n"},
    )

    errors = validate_panel(panel)

    assert _rules(errors) == {PanelRule.RAW_HEX_COLOR}
    assert hex_literal in errors[0].message


def test_var_token_only_panel_passes_hex_check(tmp_path: Path) -> None:
    html = _compliant_html(body='<div style="color: var(--text-primary);">ok</div>')
    panel = _write_panel(
        tmp_path / "panel",
        entry_html=html,
        extra_files={"panel.css": ".a { background: var(--bg-primary); }\n"},
    )

    assert validate_panel(panel) == []


def test_hex_shaped_fragment_anchor_does_not_false_positive(tmp_path: Path) -> None:
    # A fragment link with a non-hex character in its name must not be read
    # as a color -- the classic false-positive shape the pattern guards.
    html = _compliant_html(body='<a href="#section">jump</a>')
    panel = _write_panel(tmp_path / "panel", entry_html=html)

    assert validate_panel(panel) == []


def test_html_numeric_entity_does_not_false_positive(tmp_path: Path) -> None:
    # &#160; / &#8212; are numeric character references, not colors, even
    # though their digits are hex-shaped -- the (?<!&) guard must skip them.
    html = _compliant_html(body="<p>a&#160;b&#8212;c</p>")
    panel = _write_panel(tmp_path / "panel", entry_html=html)

    assert validate_panel(panel) == []


def test_wrong_length_hex_run_is_not_flagged(tmp_path: Path) -> None:
    # #12345 (5 digits) is not a valid color length; it must not be
    # mis-read as #1234 or #123.
    panel = _write_panel(
        tmp_path / "panel",
        entry_html=_compliant_html(),
        extra_files={"panel.css": ".a { --raw: #12345; }\n"},
    )

    assert validate_panel(panel) == []


# --- Aggregation + fail-closed door -------------------------------------------


def test_validate_panel_returns_all_errors_for_multiply_broken_panel(
    tmp_path: Path,
) -> None:
    # Broken three ways at once: no theme-boot script, no token stylesheet,
    # and a raw hex color. Every failure must be reported in one pass.
    html = (
        "<!DOCTYPE html>\n<html><head>\n"
        '<div style="color: #123456;">x</div>\n'
        "</head><body></body></html>\n"
    )
    panel = _write_panel(tmp_path / "panel", entry_html=html)

    errors = validate_panel(panel)

    assert PanelRule.MISSING_DESIGN_SYSTEM_LINK in _rules(errors)
    assert PanelRule.MISSING_THEME_BOOT in _rules(errors)
    assert PanelRule.RAW_HEX_COLOR in _rules(errors)


def test_assert_valid_panel_raises_bundling_every_failure(tmp_path: Path) -> None:
    html = (
        "<!DOCTYPE html>\n<html><head>\n"
        '<div style="color: #123456;">x</div>\n'
        "</head><body></body></html>\n"
    )
    panel = _write_panel(tmp_path / "panel", entry_html=html)

    with pytest.raises(PanelValidationError) as excinfo:
        assert_valid_panel(panel)

    errors = excinfo.value.errors
    assert len(errors) >= 3
    # str() lists every failure, one per line.
    rendered = str(excinfo.value)
    for error in errors:
        assert str(error) in rendered
    assert rendered.count("\n") == len(errors) - 1


def test_panel_error_str_renders_source_and_message(tmp_path: Path) -> None:
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "index.html").write_text(_compliant_html(), encoding="utf-8")

    (error,) = validate_panel(panel)

    assert str(error) == f"{error.source}: {error.message}"
    assert str(panel) in str(error)


# --- Integration: the shipped reference panel and non-compliant fixtures ------
#
# Unlike the synthetic panels above, these exercise the validator against the
# real files it exists to guard: the canonical reference panel that ships in
# the package (every future panel is copied from it, so it must always
# validate clean), and the deliberately-broken fixtures under ./fixtures/
# that prove each rule actually fires on a real directory.

# The reference panel ships inside the design_system package, next to the
# validator it must satisfy; the fixtures live beside this test file.
_REFERENCE_PANEL_DIR = Path(panels_pkg.__file__).parent / "reference"
_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_NON_COMPLIANT_FIXTURE_DIR = _FIXTURES_DIR / "non_compliant_panel"
_BAD_MANIFEST_FIXTURE_DIR = _FIXTURES_DIR / "bad_manifest_panel"


def test_reference_panel_directory_is_present() -> None:
    # A missing reference panel would make the checks below vacuously pass; a
    # standalone presence assertion keeps that failure loud and specific.
    assert (_REFERENCE_PANEL_DIR / MANIFEST_FILENAME).is_file()
    assert (_REFERENCE_PANEL_DIR / "index.html").is_file()


def test_reference_panel_validates_with_no_errors() -> None:
    assert validate_panel(_REFERENCE_PANEL_DIR) == []


def test_assert_valid_panel_does_not_raise_for_reference_panel() -> None:
    assert_valid_panel(_REFERENCE_PANEL_DIR)  # no exception


def test_reference_panel_entry_is_token_only_and_links_design_system() -> None:
    # Guard the two properties a reviewer relies on the reference to model:
    # it carries both required /design-system references, and it is entirely
    # free of raw hex color literals.
    entry_html = (_REFERENCE_PANEL_DIR / "index.html").read_text(encoding="utf-8")
    assert "/design-system/js/theme-boot.js" in entry_html
    assert "/design-system/css/tokens.css" in entry_html

    hex_errors = [
        error
        for error in validate_panel(_REFERENCE_PANEL_DIR)
        if error.rule is PanelRule.RAW_HEX_COLOR
    ]
    assert hex_errors == []


# --- Integration: the non-compliant fixtures ----------------------------------


def test_non_compliant_fixture_fails_html_rules() -> None:
    # A schema-valid manifest whose entry HTML opts out of the design system
    # and uses raw hex colors: all three HTML-level rules must fire.
    errors = validate_panel(_NON_COMPLIANT_FIXTURE_DIR)

    assert PanelRule.MISSING_DESIGN_SYSTEM_LINK in _rules(errors)
    assert PanelRule.MISSING_THEME_BOOT in _rules(errors)
    assert PanelRule.RAW_HEX_COLOR in _rules(errors)


def test_assert_valid_panel_raises_for_non_compliant_fixture() -> None:
    with pytest.raises(PanelValidationError) as excinfo:
        assert_valid_panel(_NON_COMPLIANT_FIXTURE_DIR)

    reported = {error.rule for error in excinfo.value.errors}
    assert PanelRule.RAW_HEX_COLOR in reported
    assert PanelRule.MISSING_DESIGN_SYSTEM_LINK in reported


def test_bad_manifest_fixture_fails_manifest_rule() -> None:
    # A schema-invalid manifest (non-kebab id + missing label) folds every
    # manifest failure in under MANIFEST_INVALID and short-circuits the HTML
    # checks, so that rule is the only one reported.
    errors = validate_panel(_BAD_MANIFEST_FIXTURE_DIR)

    assert _rules(errors) == {PanelRule.MANIFEST_INVALID}
    assert len(errors) >= 2  # bad id + missing label
    messages = " ".join(error.message for error in errors)
    assert "id" in messages
    assert "label" in messages


def test_assert_valid_panel_raises_for_bad_manifest_fixture() -> None:
    with pytest.raises(PanelValidationError) as excinfo:
        assert_valid_panel(_BAD_MANIFEST_FIXTURE_DIR)

    assert all(error.rule is PanelRule.MANIFEST_INVALID for error in excinfo.value.errors)
