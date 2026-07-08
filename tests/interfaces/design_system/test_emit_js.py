"""Tests for osprey.interfaces.design_system.generator.emit_js.

Builds small :class:`~osprey.interfaces.design_system.generator.model.TokenTree`
instances directly in Python (no new fixture files — this task's file
ownership is ``generator/emit_js.py`` + this test file only; ``emit_js``
only ever reads ``tree.theme_metadata``, so a full token tree is never
needed for the unit tests), plus:

- one full-pipeline test writing a tiny, valid ``tokens/`` tree to
  ``tmp_path``, loading and validating it for real, and rendering both
  artifacts end to end, and
- one read-only regression test against the real, already-committed
  ``src/osprey/interfaces/design_system/tokens/`` tree (authored by a
  separate task) confirming both renders succeed and match its real
  theme registry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import osprey.interfaces.design_system as design_system_pkg
from osprey.interfaces.design_system.generator.emit_js import (
    GENERATED_HEADER_LINES,
    STORAGE_KEY,
    ThemeManifestEntry,
    build_theme_defaults,
    build_theme_manifest,
    render_theme_boot_js,
    render_tokens_js,
)
from osprey.interfaces.design_system.generator.model import TokenTree, load_token_tree
from osprey.interfaces.design_system.generator.validate import assert_valid

REAL_TOKENS_DIR = Path(design_system_pkg.__file__).parent / "tokens"


def _tree(theme_metadata: dict[str, dict[str, object]]) -> TokenTree:
    """Build a TokenTree exposing only ``theme_metadata`` — all emit_js needs."""
    return TokenTree(
        primitives={},
        themes={},
        interfaces={},
        theme_metadata=theme_metadata,
        interface_metadata={},
    )


def _assert_hook_clean(content: str) -> None:
    """Assert generated content matches the shared determinism rules.

    ``\\n`` line endings only, no trailing whitespace on any line, and
    exactly one trailing newline — the invariants a pre-commit
    whitespace-fixer/end-of-file-fixer hook would otherwise "fix" and wedge
    the freshness gate on.
    """
    assert "\r" not in content, "must use \\n line endings, not \\r\\n"
    assert content.endswith("\n"), "must end with a trailing newline"
    assert not content.endswith("\n\n"), "must have exactly one trailing newline"
    for line in content.split("\n"):
        assert line == line.rstrip(), f"line has trailing whitespace: {line!r}"


def _exported_const(content: str, name: str) -> object:
    """Extract and JSON-parse an ``export const <name> = <json>;`` statement's value."""
    match = re.search(rf"export const {name} = (.+?);\n", content, re.DOTALL)
    assert match is not None, f"no 'export const {name} = ...;' statement found"
    return json.loads(match.group(1))


# --- build_theme_manifest --------------------------------------------------------


def test_build_theme_manifest_preserves_metadata_order() -> None:
    # Deliberately not alphabetical, to prove no internal re-sorting happens.
    tree = _tree(
        {
            "midnight": {"id": "midnight", "label": "Midnight", "mode": "dark"},
            "dawn": {"id": "dawn", "label": "Dawn", "mode": "light"},
        }
    )

    manifest = build_theme_manifest(tree)

    assert manifest == [
        ThemeManifestEntry(id="midnight", label="Midnight", mode="dark"),
        ThemeManifestEntry(id="dawn", label="Dawn", mode="light"),
    ]


def test_build_theme_manifest_empty_tree_yields_empty_manifest() -> None:
    assert build_theme_manifest(_tree({})) == []


# --- build_theme_defaults ---------------------------------------------------------


def test_build_theme_defaults_maps_each_mode_once() -> None:
    entries = [
        ThemeManifestEntry(id="dark", label="Dark", mode="dark"),
        ThemeManifestEntry(id="light", label="Light", mode="light"),
    ]

    assert build_theme_defaults(entries) == {"dark": "dark", "light": "light"}


def test_build_theme_defaults_first_entry_wins_for_a_shared_mode() -> None:
    entries = [
        ThemeManifestEntry(id="dark", label="Dark", mode="dark"),
        ThemeManifestEntry(id="midnight", label="Midnight", mode="dark"),
        ThemeManifestEntry(id="light", label="Light", mode="light"),
    ]

    assert build_theme_defaults(entries) == {"dark": "dark", "light": "light"}


def test_build_theme_defaults_omits_absent_modes() -> None:
    entries = [ThemeManifestEntry(id="light", label="Light", mode="light")]

    assert build_theme_defaults(entries) == {"light": "light"}


def test_build_theme_defaults_empty_manifest_yields_empty_defaults() -> None:
    assert build_theme_defaults([]) == {}


# --- render_tokens_js --------------------------------------------------------------


def test_render_tokens_js_starts_with_generated_header() -> None:
    content = render_tokens_js(_tree({"dark": {"id": "dark", "label": "Dark", "mode": "dark"}}))

    assert content.startswith("\n".join(GENERATED_HEADER_LINES))


def test_render_tokens_js_exports_themes_and_defaults() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark"},
            "light": {"id": "light", "label": "Light", "mode": "light"},
        }
    )

    content = render_tokens_js(tree)

    assert _exported_const(content, "THEMES") == [
        {"id": "dark", "label": "Dark", "mode": "dark"},
        {"id": "light", "label": "Light", "mode": "light"},
    ]
    assert _exported_const(content, "DEFAULTS") == {"dark": "dark", "light": "light"}


def test_render_tokens_js_carries_no_color_palettes() -> None:
    # OC-2: palettes live in theme-manager.js's computed-style bridges, not here.
    tree = _tree({"dark": {"id": "dark", "label": "Dark", "mode": "dark"}})

    content = render_tokens_js(tree)

    for forbidden in ("XTERM_PALETTES", "CHART_THEMES", "CHART_SERIES", "HLJS_THEMES", "#"):
        assert forbidden not in content


def test_render_tokens_js_is_hook_clean() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark"},
            "light": {"id": "light", "label": "Light", "mode": "light"},
        }
    )

    _assert_hook_clean(render_tokens_js(tree))


def test_render_tokens_js_is_deterministic() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark"},
            "light": {"id": "light", "label": "Light", "mode": "light"},
        }
    )

    assert render_tokens_js(tree) == render_tokens_js(tree)


def test_render_tokens_js_escapes_label_safely() -> None:
    # A label with a quote/backslash must not break out of the JSON literal.
    tree = _tree({"dark": {"id": "dark", "label": 'Dark "Mode"', "mode": "dark"}})

    content = render_tokens_js(tree)

    assert _exported_const(content, "THEMES")[0]["label"] == 'Dark "Mode"'


# --- render_theme_boot_js -----------------------------------------------------------


def _boot_globals(content: str) -> dict[str, object]:
    """Extract STORAGE_KEY/VALID_IDS/DEFAULTS literals from theme-boot.js source."""
    storage_key = re.search(r'const STORAGE_KEY = ("(?:[^"\\]|\\.)*");', content)
    valid_ids = re.search(r"const VALID_IDS = (\[.*?\]);", content)
    defaults = re.search(r"const DEFAULTS = (\{.*?\});", content, re.DOTALL)
    assert storage_key and valid_ids and defaults
    return {
        "STORAGE_KEY": json.loads(storage_key.group(1)),
        "VALID_IDS": json.loads(valid_ids.group(1)),
        "DEFAULTS": json.loads(defaults.group(1)),
    }


def test_render_theme_boot_js_starts_with_generated_header() -> None:
    content = render_theme_boot_js(_tree({"dark": {"id": "dark", "label": "Dark", "mode": "dark"}}))

    # theme-boot.js leads with an @ts-nocheck grandfather header (design_system is
    # retrofitted under the front-end type/lint gates in a later phase), followed
    # by the shared generated-file preamble.
    assert content.startswith("// @ts-nocheck\n")
    assert "\n".join(GENERATED_HEADER_LINES) in content


def test_render_theme_boot_js_is_a_classic_iife_not_a_module() -> None:
    tree = _tree({"dark": {"id": "dark", "label": "Dark", "mode": "dark"}})

    content = render_theme_boot_js(tree)
    code_lines = [line for line in content.split("\n") if not line.lstrip().startswith("//")]
    code = "\n".join(code_lines)

    assert "export " not in code
    assert not re.search(r"^\s*import\b", code, re.MULTILINE)
    assert "(function () {" in content
    assert content.rstrip("\n").endswith("})();")


def test_render_theme_boot_js_bakes_in_storage_key_and_manifest() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark"},
            "light": {"id": "light", "label": "Light", "mode": "light"},
        }
    )

    literals = _boot_globals(render_theme_boot_js(tree))

    assert literals["STORAGE_KEY"] == STORAGE_KEY == "osprey-theme"
    assert literals["VALID_IDS"] == ["dark", "light"]
    assert literals["DEFAULTS"] == {"dark": "dark", "light": "light"}


def test_render_theme_boot_js_multiline_defaults_are_reindented() -> None:
    # Regression: json.dumps(..., indent=2) embedded after "  const DEFAULTS = "
    # must have every continuation line re-indented to the surrounding
    # 2-space block, not left at column 0.
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark"},
            "light": {"id": "light", "label": "Light", "mode": "light"},
        }
    )

    content = render_theme_boot_js(tree)

    assert '  const DEFAULTS = {\n    "dark": "dark",\n    "light": "light"\n  };' in content


def test_render_theme_boot_js_reads_query_before_storage_and_falls_back_to_auto() -> None:
    # Locks the documented resolution order in the source text itself,
    # since there's no JS runtime available in this test environment to
    # execute the script (see task report: node is not installed here).
    content = render_theme_boot_js(_tree({"dark": {"id": "dark", "label": "Dark", "mode": "dark"}}))

    query_pos = content.index("readQueryTheme")
    storage_pos = content.index("readStoredTheme")
    candidate_block = content[content.index("let candidate = ") :]
    assert query_pos < storage_pos
    assert 'candidate = "auto"' in content
    assert "isKnownId(queryTheme)" in candidate_block
    assert "isKnownId(storedTheme)" in candidate_block
    assert "prefers-color-scheme: dark" in content


def test_render_theme_boot_js_is_hook_clean() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark"},
            "light": {"id": "light", "label": "Light", "mode": "light"},
        }
    )

    _assert_hook_clean(render_theme_boot_js(tree))


def test_render_theme_boot_js_is_deterministic() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark"},
            "light": {"id": "light", "label": "Light", "mode": "light"},
        }
    )

    assert render_theme_boot_js(tree) == render_theme_boot_js(tree)


# --- Full pipeline: a tiny, valid tokens/ tree on disk -----------------------------


def test_full_pipeline_renders_both_artifacts_from_a_validated_tree(tmp_path: Path) -> None:
    (tmp_path / "themes").mkdir()
    (tmp_path / "interfaces").mkdir()

    (tmp_path / "core.json").write_text(json.dumps({}), encoding="utf-8")

    dark = {
        "$extensions": {"id": "dark", "label": "Dark", "mode": "dark"},
        "bg": {"primary": {"$value": "#000000", "$type": "color"}},
        "text": {
            "primary": {"$value": "#ffffff", "$type": "color"},
            "secondary": {"$value": "#eeeeee", "$type": "color"},
            "muted": {"$value": "#aaaaaa", "$type": "color"},
        },
        "accent": {"base": {"$value": "#00ffff", "$type": "color"}},
    }
    light = {
        "$extensions": {"id": "light", "label": "Light", "mode": "light"},
        "bg": {"primary": {"$value": "#ffffff", "$type": "color"}},
        "text": {
            "primary": {"$value": "#000000", "$type": "color"},
            "secondary": {"$value": "#111111", "$type": "color"},
            "muted": {"$value": "#555555", "$type": "color"},
        },
        "accent": {"base": {"$value": "#006666", "$type": "color"}},
    }
    (tmp_path / "themes" / "dark.json").write_text(json.dumps(dark), encoding="utf-8")
    (tmp_path / "themes" / "light.json").write_text(json.dumps(light), encoding="utf-8")

    tree = load_token_tree(tmp_path)
    assert_valid(tree)

    tokens_js = render_tokens_js(tree)
    boot_js = render_theme_boot_js(tree)

    assert _exported_const(tokens_js, "THEMES") == [
        {"id": "dark", "label": "Dark", "mode": "dark"},
        {"id": "light", "label": "Light", "mode": "light"},
    ]
    assert _exported_const(tokens_js, "DEFAULTS") == {"dark": "dark", "light": "light"}
    literals = _boot_globals(boot_js)
    assert literals["VALID_IDS"] == ["dark", "light"]
    assert literals["DEFAULTS"] == {"dark": "dark", "light": "light"}
    _assert_hook_clean(tokens_js)
    _assert_hook_clean(boot_js)


# --- Regression: the real, already-committed tokens/ tree -------------------------


@pytest.mark.skipif(not REAL_TOKENS_DIR.is_dir(), reason="real tokens/ tree not present yet")
def test_real_tokens_tree_renders_both_artifacts_cleanly() -> None:
    tree = load_token_tree(REAL_TOKENS_DIR)
    assert_valid(tree)

    tokens_js = render_tokens_js(tree)
    boot_js = render_theme_boot_js(tree)

    themes = _exported_const(tokens_js, "THEMES")
    assert {entry["id"] for entry in themes} == {"dark", "light"}
    assert {entry["mode"] for entry in themes} == {"dark", "light"}
    assert _exported_const(tokens_js, "DEFAULTS") == {"dark": "dark", "light": "light"}

    literals = _boot_globals(boot_js)
    assert set(literals["VALID_IDS"]) == {"dark", "light"}
    assert literals["DEFAULTS"] == {"dark": "dark", "light": "light"}

    _assert_hook_clean(tokens_js)
    _assert_hook_clean(boot_js)
