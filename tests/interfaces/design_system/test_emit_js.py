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
    ThemeFamilyDefaultsError,
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
            "midnight": {"id": "midnight", "label": "Midnight", "mode": "dark", "family": "osprey"},
            "dawn": {"id": "dawn", "label": "Dawn", "mode": "light", "family": "osprey"},
        }
    )

    manifest = build_theme_manifest(tree)

    assert manifest == [
        ThemeManifestEntry(id="midnight", label="Midnight", mode="dark", family="osprey"),
        ThemeManifestEntry(id="dawn", label="Dawn", mode="light", family="osprey"),
    ]


def test_build_theme_manifest_reads_family_from_metadata() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "hc-dark": {
                "id": "hc-dark",
                "label": "HC Dark",
                "mode": "dark",
                "family": "high-contrast",
            },
        }
    )

    manifest = build_theme_manifest(tree)

    assert [entry.family for entry in manifest] == ["osprey", "high-contrast"]


def test_build_theme_manifest_empty_tree_yields_empty_manifest() -> None:
    assert build_theme_manifest(_tree({})) == []


# --- build_theme_defaults ---------------------------------------------------------


def test_build_theme_defaults_groups_by_family() -> None:
    entries = [
        ThemeManifestEntry(id="dark", label="Dark", mode="dark", family="osprey"),
        ThemeManifestEntry(id="light", label="Light", mode="light", family="osprey"),
    ]

    assert build_theme_defaults(entries) == {"osprey": {"dark": "dark", "light": "light"}}


def test_build_theme_defaults_keeps_families_independent() -> None:
    # Two families sharing a mode must not collide with each other — each
    # family gets its own {mode: id} entry, keyed by its declared family,
    # never by lexical/manifest order across families.
    entries = [
        ThemeManifestEntry(id="dark", label="Dark", mode="dark", family="osprey"),
        ThemeManifestEntry(id="light", label="Light", mode="light", family="osprey"),
        ThemeManifestEntry(id="hc-dark", label="HC Dark", mode="dark", family="high-contrast"),
        ThemeManifestEntry(id="hc-light", label="HC Light", mode="light", family="high-contrast"),
    ]

    assert build_theme_defaults(entries) == {
        "osprey": {"dark": "dark", "light": "light"},
        "high-contrast": {"dark": "hc-dark", "light": "hc-light"},
    }


def test_build_theme_defaults_ambiguous_mode_within_family_raises() -> None:
    # A new theme file sharing a family+mode with an existing one must fail
    # closed instead of silently winning/losing by manifest order (skeptic
    # finding M2 — this is the case the old first-filename-wins logic let
    # through).
    entries = [
        ThemeManifestEntry(id="dark", label="Dark", mode="dark", family="osprey"),
        ThemeManifestEntry(id="midnight", label="Midnight", mode="dark", family="osprey"),
        ThemeManifestEntry(id="light", label="Light", mode="light", family="osprey"),
    ]

    with pytest.raises(ThemeFamilyDefaultsError, match="osprey"):
        build_theme_defaults(entries)


def test_build_theme_defaults_missing_mode_in_family_raises() -> None:
    # A family with only a light theme (no dark counterpart) can't produce
    # a complete auto default for that family — fail closed rather than
    # silently omitting the missing mode.
    entries = [ThemeManifestEntry(id="light", label="Light", mode="light", family="osprey")]

    with pytest.raises(ThemeFamilyDefaultsError, match="osprey"):
        build_theme_defaults(entries)


def test_build_theme_defaults_empty_manifest_yields_empty_defaults() -> None:
    assert build_theme_defaults([]) == {}


# --- render_tokens_js --------------------------------------------------------------


def test_render_tokens_js_starts_with_generated_header() -> None:
    content = render_tokens_js(
        _tree(
            {
                "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
                "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
            }
        )
    )

    assert content.startswith("\n".join(GENERATED_HEADER_LINES))


def test_render_tokens_js_exports_themes_and_defaults() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

    content = render_tokens_js(tree)

    assert _exported_const(content, "THEMES") == [
        {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
        {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
    ]
    assert _exported_const(content, "DEFAULTS") == {"osprey": {"dark": "dark", "light": "light"}}


def test_render_tokens_js_exports_default_family_matching_theme_boot() -> None:
    # tokens.js and theme-boot.js must never disagree on the fallback
    # family: both derive it from the same _default_family() helper (the
    # first family declared in the manifest).
    tree = _tree(
        {
            "hc-dark": {
                "id": "hc-dark",
                "label": "HC Dark",
                "mode": "dark",
                "family": "high-contrast",
            },
            "hc-light": {
                "id": "hc-light",
                "label": "HC Light",
                "mode": "light",
                "family": "high-contrast",
            },
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

    tokens_content = render_tokens_js(tree)
    boot_content = render_theme_boot_js(tree)

    assert _exported_const(tokens_content, "DEFAULT_FAMILY") == "high-contrast"
    assert _boot_globals(boot_content)["DEFAULT_FAMILY"] == "high-contrast"


def test_render_tokens_js_themes_entries_carry_family() -> None:
    # THEMES must expose each theme's family so the runtime/switcher can
    # group by family, not just by mode.
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
            "hc-dark": {
                "id": "hc-dark",
                "label": "HC Dark",
                "mode": "dark",
                "family": "high-contrast",
            },
            "hc-light": {
                "id": "hc-light",
                "label": "HC Light",
                "mode": "light",
                "family": "high-contrast",
            },
        }
    )

    content = render_tokens_js(tree)

    themes = _exported_const(content, "THEMES")
    assert all("family" in entry for entry in themes)
    assert {entry["family"] for entry in themes} == {"osprey", "high-contrast"}


def test_render_tokens_js_carries_no_color_palettes() -> None:
    # OC-2: palettes live in theme-manager.js's computed-style bridges, not here.
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

    content = render_tokens_js(tree)

    for forbidden in ("XTERM_PALETTES", "CHART_THEMES", "CHART_SERIES", "HLJS_THEMES", "#"):
        assert forbidden not in content


def test_render_tokens_js_is_hook_clean() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

    _assert_hook_clean(render_tokens_js(tree))


def test_render_tokens_js_is_deterministic() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

    assert render_tokens_js(tree) == render_tokens_js(tree)


def test_render_tokens_js_escapes_label_safely() -> None:
    # A label with a quote/backslash must not break out of the JSON literal.
    tree = _tree(
        {
            "dark": {"id": "dark", "label": 'Dark "Mode"', "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

    content = render_tokens_js(tree)

    assert _exported_const(content, "THEMES")[0]["label"] == 'Dark "Mode"'


# --- render_theme_boot_js -----------------------------------------------------------


def _boot_globals(content: str) -> dict[str, object]:
    """Extract baked-literal globals from theme-boot.js.

    STORAGE_KEY/VALID_IDS/DEFAULTS/FAMILY_BY_ID/DEFAULT_FAMILY.
    """
    storage_key = re.search(r'const STORAGE_KEY = ("(?:[^"\\]|\\.)*");', content)
    valid_ids = re.search(r"const VALID_IDS = (\[.*?\]);", content)
    defaults = re.search(r"const DEFAULTS = (\{.*?\});", content, re.DOTALL)
    family_by_id = re.search(r"const FAMILY_BY_ID = (\{.*?\});", content, re.DOTALL)
    default_family = re.search(r"const DEFAULT_FAMILY = (.*?);", content)
    assert storage_key and valid_ids and defaults and family_by_id and default_family
    return {
        "STORAGE_KEY": json.loads(storage_key.group(1)),
        "VALID_IDS": json.loads(valid_ids.group(1)),
        "DEFAULTS": json.loads(defaults.group(1)),
        "FAMILY_BY_ID": json.loads(family_by_id.group(1)),
        "DEFAULT_FAMILY": json.loads(default_family.group(1)),
    }


def test_render_theme_boot_js_starts_with_generated_header() -> None:
    content = render_theme_boot_js(
        _tree(
            {
                "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
                "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
            }
        )
    )

    # theme-boot.js leads with an @ts-check header — its emitted isKnownId type
    # predicate makes it strict-clean under checkJs — followed by the shared
    # generated-file preamble.
    assert content.startswith("// @ts-check\n")
    assert "\n".join(GENERATED_HEADER_LINES) in content


def test_render_theme_boot_js_is_a_classic_iife_not_a_module() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

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
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

    literals = _boot_globals(render_theme_boot_js(tree))

    assert literals["STORAGE_KEY"] == STORAGE_KEY == "osprey-theme"
    assert literals["VALID_IDS"] == ["dark", "light"]
    assert literals["DEFAULTS"] == {"osprey": {"dark": "dark", "light": "light"}}
    assert literals["FAMILY_BY_ID"] == {"dark": "osprey", "light": "osprey"}
    assert literals["DEFAULT_FAMILY"] == "osprey"


def test_render_theme_boot_js_default_family_is_first_declared_family() -> None:
    # With more than one family, DEFAULT_FAMILY (the fallback used only
    # when no valid server data-theme attribute is present — see the
    # render_theme_boot_js docstring) picks the first family in manifest
    # order. FAMILY_BY_ID separately maps every id to its own declared
    # family, regardless of DEFAULT_FAMILY.
    tree = _tree(
        {
            "hc-dark": {
                "id": "hc-dark",
                "label": "HC Dark",
                "mode": "dark",
                "family": "high-contrast",
            },
            "hc-light": {
                "id": "hc-light",
                "label": "HC Light",
                "mode": "light",
                "family": "high-contrast",
            },
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

    literals = _boot_globals(render_theme_boot_js(tree))

    assert literals["DEFAULT_FAMILY"] == "high-contrast"
    assert literals["FAMILY_BY_ID"] == {
        "hc-dark": "high-contrast",
        "hc-light": "high-contrast",
        "dark": "osprey",
        "light": "osprey",
    }
    assert literals["DEFAULTS"] == {
        "high-contrast": {"dark": "hc-dark", "light": "hc-light"},
        "osprey": {"dark": "dark", "light": "light"},
    }


def test_render_theme_boot_js_multiline_defaults_are_reindented() -> None:
    # Regression: json.dumps(..., indent=2) embedded after "  const DEFAULTS = "
    # must have every continuation line re-indented to the surrounding
    # 2-space block, not left at column 0.
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

    content = render_theme_boot_js(tree)

    assert (
        "  const DEFAULTS = {\n"
        '    "osprey": {\n'
        '      "dark": "dark",\n'
        '      "light": "light"\n'
        "    }\n"
        "  };"
    ) in content


def test_render_theme_boot_js_reads_query_before_storage_before_server_before_auto() -> None:
    # Locks the documented resolution order in the source text itself,
    # since there's no JS runtime available in this test environment to
    # execute the script (see task report: node is not installed here).
    content = render_theme_boot_js(
        _tree(
            {
                "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
                "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
            }
        )
    )

    query_pos = content.index("readQueryTheme")
    storage_pos = content.index("readStoredTheme")
    server_pos = content.index("readServerTheme")
    candidate_block = content[content.index("let candidate = ") :]
    query_check_pos = candidate_block.index("isKnownId(queryTheme)")
    storage_check_pos = candidate_block.index("isKnownId(storedTheme)")
    server_check_pos = candidate_block.index("isValidId(serverTheme)")
    assert query_pos < storage_pos < server_pos
    assert query_check_pos < storage_check_pos < server_check_pos
    assert 'candidate = "auto"' in content
    assert "prefers-color-scheme: dark" in content


def test_render_theme_boot_js_reads_server_attr_from_html_data_theme() -> None:
    # Server-attr contract: document.documentElement.getAttribute("data-theme"),
    # i.e. the data-theme attribute on <html> — this is what Task 1.10's
    # server render must produce.
    content = render_theme_boot_js(
        _tree(
            {
                "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
                "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
            }
        )
    )

    assert 'document.documentElement.getAttribute("data-theme")' in content


def test_render_theme_boot_js_server_rung_uses_isvalidid_not_isknownid() -> None:
    # The server attribute is a concrete theme id the server resolved from
    # config, never the literal "auto" — so the candidate rung and the
    # family-lookup guard both gate on isValidId(serverTheme), not the
    # looser isKnownId (which also accepts "auto") used for query/storage.
    content = render_theme_boot_js(
        _tree(
            {
                "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
                "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
            }
        )
    )

    assert "isKnownId(serverTheme)" not in content
    assert content.count("isValidId(serverTheme)") == 2  # familyForAuto + candidate rung


def test_render_theme_boot_js_family_for_auto_prefers_server_theme_family() -> None:
    # The family 'auto' resolves within comes from FAMILY_BY_ID[serverTheme]
    # when the server attr is a valid concrete id, else DEFAULT_FAMILY.
    content = render_theme_boot_js(
        _tree(
            {
                "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
                "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
            }
        )
    )

    assert (
        "const familyForAuto = isValidId(serverTheme) ? FAMILY_BY_ID[serverTheme] : DEFAULT_FAMILY;"
    ) in content
    assert "resolveAuto(familyForAuto)" in content


def test_render_theme_boot_js_does_not_unconditionally_clobber_server_attr() -> None:
    # No-FOUC contract (finding I4): the final setAttribute call must be
    # gated on the resolved id actually differing from what the server
    # already rendered, not fire unconditionally on every load.
    content = render_theme_boot_js(
        _tree(
            {
                "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
                "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
            }
        )
    )

    assert "if (resolved && resolved !== serverTheme) {" in content
    assert content.count('document.documentElement.setAttribute("data-theme"') == 1
    # The only setAttribute call is the guarded one above (no earlier,
    # unconditional write to clobber a correct server-rendered attribute).
    unconditional_write = re.search(
        r'^\s*document\.documentElement\.setAttribute\("data-theme"', content, re.MULTILINE
    )
    assert unconditional_write is not None
    guarded_line_start = content.index("if (resolved && resolved !== serverTheme)")
    assert unconditional_write.start() > guarded_line_start


def test_render_theme_boot_js_is_hook_clean() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

    _assert_hook_clean(render_theme_boot_js(tree))


def test_render_theme_boot_js_is_deterministic() -> None:
    tree = _tree(
        {
            "dark": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
            "light": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        }
    )

    assert render_theme_boot_js(tree) == render_theme_boot_js(tree)


# --- Full pipeline: a tiny, valid tokens/ tree on disk -----------------------------


def test_full_pipeline_renders_both_artifacts_from_a_validated_tree(tmp_path: Path) -> None:
    (tmp_path / "themes").mkdir()
    (tmp_path / "interfaces").mkdir()

    (tmp_path / "core.json").write_text(json.dumps({}), encoding="utf-8")

    dark = {
        "$extensions": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
        "bg": {"primary": {"$value": "#000000", "$type": "color"}},
        "text": {
            "primary": {"$value": "#ffffff", "$type": "color"},
            "secondary": {"$value": "#eeeeee", "$type": "color"},
            "muted": {"$value": "#aaaaaa", "$type": "color"},
        },
        "accent": {"base": {"$value": "#00ffff", "$type": "color"}},
    }
    light = {
        "$extensions": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
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
        {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
        {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
    ]
    assert _exported_const(tokens_js, "DEFAULTS") == {"osprey": {"dark": "dark", "light": "light"}}
    literals = _boot_globals(boot_js)
    assert literals["VALID_IDS"] == ["dark", "light"]
    assert literals["DEFAULTS"] == {"osprey": {"dark": "dark", "light": "light"}}
    assert literals["FAMILY_BY_ID"] == {"dark": "osprey", "light": "osprey"}
    assert literals["DEFAULT_FAMILY"] == "osprey"
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
    assert {entry["id"] for entry in themes} == {
        "dark",
        "light",
        "high-contrast-dark",
        "high-contrast-light",
    }
    assert {entry["mode"] for entry in themes} == {"dark", "light"}
    assert {entry["family"] for entry in themes} == {"osprey", "high-contrast"}
    assert _exported_const(tokens_js, "DEFAULTS") == {
        "osprey": {"dark": "dark", "light": "light"},
        "high-contrast": {"dark": "high-contrast-dark", "light": "high-contrast-light"},
    }

    literals = _boot_globals(boot_js)
    assert set(literals["VALID_IDS"]) == {
        "dark",
        "light",
        "high-contrast-dark",
        "high-contrast-light",
    }
    assert literals["DEFAULTS"] == {
        "osprey": {"dark": "dark", "light": "light"},
        "high-contrast": {"dark": "high-contrast-dark", "light": "high-contrast-light"},
    }
    assert literals["FAMILY_BY_ID"] == {
        "dark": "osprey",
        "light": "osprey",
        "high-contrast-dark": "high-contrast",
        "high-contrast-light": "high-contrast",
    }
    assert literals["DEFAULT_FAMILY"] == "osprey"

    _assert_hook_clean(tokens_js)
    _assert_hook_clean(boot_js)
