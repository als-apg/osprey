"""Contract tests: invariants the REAL design-token sources must hold.

Unlike ``test_model.py``/``test_validate.py``/``test_emit_*.py``/``test_build.py``
(hermetic, fixture- or ``tmp_path``-based), every test in this file runs
against the actual, already-committed ``tokens/`` tree — this is the
top-level data-quality gate for the shipped tokens, not the generator code.

Covers the full generator validation sweep (re-run here against real data
as the authoritative contract, reusing ``generator/validate.py``'s checks
per the design spec's explicit "reused by contract tests" directive),
``code.*`` values against the real vendor manifest, ``tokens.js``'s THEMES
manifest / DEFAULTS parity with the theme sources, and two additional
checks carried forward from a design review: no orphan color primitives,
and no literal color value that duplicates a ramp step where a one-hop
alias would do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import osprey.interfaces as interfaces_pkg
import osprey.interfaces.design_system as design_system_pkg
from osprey.interfaces.design_system.generator.emit_js import (
    build_theme_defaults,
    build_theme_manifest,
)
from osprey.interfaces.design_system.generator.model import AliasStatus, TokenTree, load_token_tree
from osprey.interfaces.design_system.generator.validate import (
    check_alias_resolution,
    check_color_syntax,
    check_interface_mode_completeness,
    check_namespace_collisions,
    check_terminal_serialization,
    check_theme_completeness,
    check_theme_metadata,
    check_wcag_gates,
    parse_color,
    validate_token_tree,
)

TOKENS_DIR = Path(design_system_pkg.__file__).parent / "tokens"
VENDOR_MANIFEST_PATH = Path(interfaces_pkg.__file__).parent / "vendor_manifest.json"


@pytest.fixture(scope="module")
def tree() -> TokenTree:
    """The real, already-committed token tree — loaded once per test module."""
    return load_token_tree(TOKENS_DIR)


# --- Full validator sweep, re-run here as the authoritative real-data gate -------


def test_real_tree_passes_every_generator_validation_check(tree: TokenTree) -> None:
    errors = validate_token_tree(tree)
    assert errors == [], "\n".join(str(error) for error in errors)


# Individual checks asserted explicitly too (per the plan's enumerated list),
# so a failure here names exactly which contract broke without needing to
# read generator/validate.py's combined sweep output.


def test_every_theme_defines_identical_token_set(tree: TokenTree) -> None:
    assert check_theme_completeness(tree) == []


def test_every_interface_mode_group_matches_its_themes(tree: TokenTree) -> None:
    assert check_interface_mode_completeness(tree) == []


def test_every_color_value_parses(tree: TokenTree) -> None:
    assert check_color_syntax(tree) == []


def test_wcag_gates_pass_every_theme(tree: TokenTree) -> None:
    assert check_wcag_gates(tree) == []


def test_terminal_group_values_are_xterm_safe(tree: TokenTree) -> None:
    assert check_terminal_serialization(tree) == []


def test_no_dangling_or_multi_hop_aliases(tree: TokenTree) -> None:
    assert check_alias_resolution(tree) == []


def test_no_extension_semantic_namespace_collisions(tree: TokenTree) -> None:
    assert check_namespace_collisions(tree) == []


def test_theme_metadata_is_well_formed(tree: TokenTree) -> None:
    assert check_theme_metadata(tree) == []


# --- code.* values exist as vendor_manifest.json asset names ---------------------


def test_code_theme_values_are_known_vendor_assets(tree: TokenTree) -> None:
    manifest = json.loads(VENDOR_MANIFEST_PATH.read_text(encoding="utf-8"))
    asset_names = {asset["name"] for asset in manifest["assets"]}

    for stem, tokens in tree.themes.items():
        token = tokens.get("code.theme")
        assert token is not None, f"theme {stem!r} has no code.theme token"
        assert token.value in asset_names, (
            f"theme {stem!r} code.theme={token.value!r} does not match any "
            f"asset name in {VENDOR_MANIFEST_PATH}"
        )


# --- tokens.js THEMES manifest / DEFAULTS parity with the theme sources ----------


def test_theme_manifest_ids_match_theme_files(tree: TokenTree) -> None:
    entries = build_theme_manifest(tree)

    manifest_ids = {entry.id for entry in entries}
    theme_file_ids = {metadata["id"] for metadata in tree.theme_metadata.values()}
    assert manifest_ids == theme_file_ids


def test_theme_manifest_labels_and_modes_match_theme_files(tree: TokenTree) -> None:
    entries = {entry.id: entry for entry in build_theme_manifest(tree)}

    for stem, metadata in tree.theme_metadata.items():
        entry = entries[metadata["id"]]
        assert entry.label == metadata["label"], stem
        assert entry.mode == metadata["mode"], stem


def test_theme_defaults_ids_exist_with_correct_modes(tree: TokenTree) -> None:
    entries = build_theme_manifest(tree)
    defaults = build_theme_defaults(entries)
    entries_by_id = {entry.id: entry for entry in entries}

    # Every mode actually present among the themes must have a default.
    assert set(defaults) == {entry.mode for entry in entries}
    for mode, theme_id in defaults.items():
        assert theme_id in entries_by_id, (
            f"DEFAULTS[{mode!r}] = {theme_id!r} is not a known theme id"
        )
        assert entries_by_id[theme_id].mode == mode


# --- Review #2 addition 1: no orphan color ramp primitives -----------------------
#
# Scoped to color.* specifically: core.json's space/radius/z/duration/text
# (type-scale) primitives are legitimately never alias-referenced by design
# (see core.json's own $description — they're a forward-looking scale with
# no consuming token yet), and font.display/font.mono are deliberately
# promoted straight to CSS by emit_css.py rather than aliased from a
# semantic token. Only unreferenced *color* ramp steps are a genuine
# "why does this exist" smell — this is what would have caught the
# amber.250 orphan (fixed by aliasing lat-led.computing to it).


def test_every_color_primitive_has_at_least_one_referrer(tree: TokenTree) -> None:
    referenced_paths: set[str] = set()
    for tokens in (*tree.themes.values(), *tree.interfaces.values(), tree.primitives):
        for token in tokens.values():
            if token.alias_status == AliasStatus.RESOLVED:
                referenced_paths.add(token.alias_target)  # type: ignore[arg-type]

    color_primitives = {path for path in tree.primitives if path.startswith("color.")}
    orphans = sorted(color_primitives - referenced_paths)
    assert orphans == [], f"orphan color primitives (no alias references them): {orphans}"


# --- Review #2 addition 2: literal opaque colors must not duplicate a ramp step --
#
# A theme/interface token whose *fully opaque* (alpha == 1.0) value equals a
# color primitive's value could have been expressed as a one-hop alias
# instead of a hand-copied literal, and should be — that's the "literal/ramp
# drift" this catches. Alpha composites (alpha < 1.0, e.g. tint.*/border
# alpha overlays) are exempt: no ramp step encodes "primitive + alpha", so a
# one-hop alias can't express them (see dark.json/light.json's own
# per-token $description comments) — this is not a loophole, it's the
# documented reason those specific tokens are allowed to stay literal.
#
# (scope, dot-path) pairs codified here as genuinely acceptable literals even
# though they're fully opaque, per the design review:
_ALLOWED_OPAQUE_LITERAL_DUPLICATES: frozenset[tuple[str, str]] = frozenset(
    {
        # lat-led.idle (#4a5264) has no matching slate ramp step — verified
        # by the review; listed here defensively in case a future core.json
        # addition happens to coincide with it.
        ("interfaces/lattice_dashboard", "dark.lat-led.idle"),
        ("interfaces/lattice_dashboard", "light.lat-led.idle"),
        # light.json chart.grid is net-new by design (tuning had no
        # light-mode plot layout at all — the dark-locked-plots bug this
        # feature fixes), not copied from an existing ramp step.
        ("themes/light", "chart.grid"),
    }
)


def test_literal_opaque_colors_do_not_duplicate_a_ramp_step(tree: TokenTree) -> None:
    primitives_by_rgb: dict[tuple[int, int, int], list[str]] = {}
    for path, token in tree.primitives.items():
        if not path.startswith("color.") or not isinstance(token.value, str):
            continue
        color = parse_color(token.value)
        if color is not None and color.alpha == 1.0:
            primitives_by_rgb.setdefault((color.red, color.green, color.blue), []).append(path)

    violations: list[str] = []
    for scope_name, scope in (("themes", tree.themes), ("interfaces", tree.interfaces)):
        for stem, tokens in scope.items():
            scope_key = f"{scope_name}/{stem}"
            for path, token in tokens.items():
                if token.alias_status != AliasStatus.NOT_ALIAS:
                    continue  # aliases (resolved or not) aren't literals
                if (scope_key, path) in _ALLOWED_OPAQUE_LITERAL_DUPLICATES:
                    continue
                if not isinstance(token.value, str):
                    continue
                color = parse_color(token.value)
                if color is None or color.alpha != 1.0:
                    continue
                match = primitives_by_rgb.get((color.red, color.green, color.blue))
                if match:
                    violations.append(
                        f"{scope_key}:{path} = {token.value!r} duplicates primitive(s) "
                        f"{match} — express as a one-hop alias instead of a literal"
                    )
    assert violations == [], "\n".join(violations)
