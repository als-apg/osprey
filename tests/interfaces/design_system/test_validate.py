"""Tests for osprey.interfaces.design_system.generator.validate.

Builds small :class:`~osprey.interfaces.design_system.generator.model.TokenTree`
instances directly in Python (no new fixture files — this task's file
ownership is ``generator/validate.py`` + this test file only) to exercise
each check in isolation, plus:

- one full-pipeline test writing a tiny, valid ``tokens/`` tree to
  ``tmp_path`` and confirming it validates clean end to end, and
- one read-only regression test against the real, already-committed
  ``src/osprey/interfaces/design_system/tokens/`` tree (authored by a
  separate task) confirming it currently validates clean.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import osprey.interfaces.design_system as design_system_pkg
from osprey.interfaces.design_system.generator.model import (
    AliasStatus,
    ResolvedToken,
    TokenTree,
    load_token_tree,
)
from osprey.interfaces.design_system.generator.validate import (
    WCAG_GATES,
    WCAG_GATES_AAA,
    RGBColor,
    TokenValidationError,
    ValidationRule,
    assert_valid,
    check_alias_resolution,
    check_color_syntax,
    check_interface_mode_completeness,
    check_namespace_collisions,
    check_promoted_primitive_collisions,
    check_terminal_serialization,
    check_theme_completeness,
    check_theme_metadata,
    check_wcag_gates,
    contrast_ratio,
    gates_for_family,
    is_terminal_safe_color,
    parse_color,
    relative_luminance,
    validate_token_tree,
)

REAL_TOKENS_DIR = Path(design_system_pkg.__file__).parent / "tokens"


def _token(
    path: str,
    value: object,
    *,
    type_: str | None = "color",
    alias_status: AliasStatus = AliasStatus.NOT_ALIAS,
    alias_target: str | None = None,
    source_file: Path = Path("<test>"),
) -> ResolvedToken:
    """Build a minimal ResolvedToken for check_* unit tests."""
    return ResolvedToken(
        path=path,
        value=value,
        type=type_,
        description=None,
        extensions={},
        source_file=source_file,
        alias_status=alias_status,
        alias_target=alias_target,
    )


def _tree(
    *,
    primitives: dict[str, ResolvedToken] | None = None,
    themes: dict[str, dict[str, ResolvedToken]] | None = None,
    interfaces: dict[str, dict[str, ResolvedToken]] | None = None,
    theme_metadata: dict[str, dict[str, object]] | None = None,
    interface_metadata: dict[str, dict[str, object]] | None = None,
) -> TokenTree:
    """Build a TokenTree with empty defaults for whatever isn't under test."""
    return TokenTree(
        primitives=primitives or {},
        themes=themes or {},
        interfaces=interfaces or {},
        theme_metadata=theme_metadata or {},
        interface_metadata=interface_metadata or {},
    )


# --- parse_color / is_terminal_safe_color --------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#fff", RGBColor(255, 255, 255, 1.0)),
        ("#000", RGBColor(0, 0, 0, 1.0)),
        ("#0f08", RGBColor(0, 255, 0, 136 / 255)),
        ("#14b8a6", RGBColor(0x14, 0xB8, 0xA6, 1.0)),
        ("#14b8a680", RGBColor(0x14, 0xB8, 0xA6, 0x80 / 255)),
        ("rgba(148, 163, 184, 0.12)", RGBColor(148, 163, 184, 0.12)),
        ("rgb(148, 163, 184)", RGBColor(148, 163, 184, 1.0)),
        ("rgba(239, 68, 68, 1.0)", RGBColor(239, 68, 68, 1.0)),
    ],
)
def test_parse_color_accepts_valid_syntax(value: str, expected: RGBColor) -> None:
    assert parse_color(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "hsl(120, 50%, 50%)",
        "rgb(0 0 0 / 50%)",
        "not-a-color",
        "#gggggg",
        "#12345",
        "rgba(256, 0, 0, 1)",
        "rgba(0, 0, 0, 1.5)",
        "{alias.path}",
        "",
    ],
)
def test_parse_color_rejects_invalid_syntax(value: str) -> None:
    assert parse_color(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#14b8a6", True),
        ("#14b8a680", True),
        ("rgba(148, 163, 184, 0.12)", True),
        ("rgb(148, 163, 184)", True),
        ("#fff", False),  # short hex not accepted for terminal group
        ("#f0f8", False),
        ("hsl(120, 50%, 50%)", False),
        ("not-a-color", False),
    ],
)
def test_is_terminal_safe_color(value: str, expected: bool) -> None:
    assert is_terminal_safe_color(value) is expected


# --- relative_luminance / contrast_ratio ----------------------------------------


def test_contrast_ratio_black_on_white_is_max() -> None:
    ratio = contrast_ratio(RGBColor(0, 0, 0), RGBColor(255, 255, 255))
    assert ratio == pytest.approx(21.0, abs=0.01)


def test_contrast_ratio_identical_colors_is_one() -> None:
    color = RGBColor(100, 100, 100)
    assert contrast_ratio(color, color) == pytest.approx(1.0, abs=1e-9)


def test_contrast_ratio_is_symmetric() -> None:
    a, b = RGBColor(10, 20, 30), RGBColor(200, 210, 220)
    assert contrast_ratio(a, b) == pytest.approx(contrast_ratio(b, a))


def test_relative_luminance_black_and_white() -> None:
    assert relative_luminance(RGBColor(0, 0, 0)) == pytest.approx(0.0, abs=1e-9)
    assert relative_luminance(RGBColor(255, 255, 255)) == pytest.approx(1.0, abs=1e-9)


# --- check_alias_resolution -----------------------------------------------------


@pytest.mark.parametrize(
    ("status", "rule"),
    [
        (AliasStatus.DANGLING, ValidationRule.DANGLING_ALIAS),
        (AliasStatus.MULTI_HOP, ValidationRule.MULTI_HOP_ALIAS),
        (AliasStatus.NOT_PRIMITIVE, ValidationRule.NOT_PRIMITIVE_ALIAS),
    ],
)
def test_check_alias_resolution_flags_unresolved_statuses(
    status: AliasStatus, rule: ValidationRule
) -> None:
    tree = _tree(
        themes={
            "dark": {
                "accent.base": _token(
                    "accent.base",
                    "{some.target}",
                    alias_status=status,
                    alias_target="some.target",
                    source_file=Path("themes/dark.json"),
                )
            }
        }
    )

    errors = check_alias_resolution(tree)

    assert len(errors) == 1
    assert errors[0].rule == rule
    assert errors[0].path == "accent.base"
    assert errors[0].source_file == Path("themes/dark.json")


def test_check_alias_resolution_ignores_resolved_and_literal() -> None:
    tree = _tree(
        primitives={"color.a": _token("color.a", "#000000")},
        themes={
            "dark": {
                "bg.primary": _token("bg.primary", "#000000", alias_status=AliasStatus.NOT_ALIAS),
                "accent.base": _token(
                    "accent.base",
                    "#111111",
                    alias_status=AliasStatus.RESOLVED,
                    alias_target="color.a",
                ),
            }
        },
    )

    assert check_alias_resolution(tree) == []


# --- check_color_syntax ----------------------------------------------------------


def test_check_color_syntax_flags_invalid_color_value() -> None:
    tree = _tree(themes={"dark": {"bg.primary": _token("bg.primary", "not-a-color")}})

    errors = check_color_syntax(tree)

    assert len(errors) == 1
    assert errors[0].rule == ValidationRule.INVALID_COLOR
    assert errors[0].path == "bg.primary"


def test_check_color_syntax_accepts_valid_color_value() -> None:
    tree = _tree(themes={"dark": {"bg.primary": _token("bg.primary", "#0a0f1a")}})

    assert check_color_syntax(tree) == []


def test_check_color_syntax_skips_non_color_type() -> None:
    tree = _tree(
        themes={"dark": {"shadow.panel": _token("shadow.panel", "not-a-color", type_="shadow")}}
    )

    assert check_color_syntax(tree) == []


def test_check_color_syntax_skips_unresolved_aliases() -> None:
    # Already reported by check_alias_resolution; would be double-counted
    # (and misleadingly parsed as "not a color") otherwise.
    tree = _tree(
        themes={
            "dark": {
                "accent.base": _token(
                    "accent.base",
                    "{missing.target}",
                    alias_status=AliasStatus.DANGLING,
                    alias_target="missing.target",
                )
            }
        }
    )

    assert check_color_syntax(tree) == []


# --- check_terminal_serialization -------------------------------------------------


def test_check_terminal_serialization_rejects_short_hex() -> None:
    tree = _tree(themes={"dark": {"terminal.ansi.red": _token("terminal.ansi.red", "#f00")}})

    errors = check_terminal_serialization(tree)

    assert len(errors) == 1
    assert errors[0].rule == ValidationRule.TERMINAL_SERIALIZATION
    assert errors[0].path == "terminal.ansi.red"


def test_check_terminal_serialization_accepts_full_hex_and_legacy_rgba() -> None:
    tree = _tree(
        themes={
            "dark": {
                "terminal.text": _token("terminal.text", "#c8d6e5"),
                "terminal.selection": _token("terminal.selection", "rgba(79, 209, 197, 0.25)"),
            }
        }
    )

    assert check_terminal_serialization(tree) == []


def test_check_terminal_serialization_ignores_non_terminal_paths() -> None:
    # Short hex is valid CSS and fine outside the terminal group.
    tree = _tree(themes={"dark": {"bg.primary": _token("bg.primary", "#fff")}})

    assert check_terminal_serialization(tree) == []


# --- check_theme_completeness -----------------------------------------------------


def test_check_theme_completeness_flags_missing_token() -> None:
    tree = _tree(
        themes={
            "dark": {
                "bg.primary": _token("bg.primary", "#000", source_file=Path("themes/dark.json")),
                "text.primary": _token(
                    "text.primary", "#fff", source_file=Path("themes/dark.json")
                ),
            },
            "light": {
                "bg.primary": _token("bg.primary", "#fff", source_file=Path("themes/light.json")),
            },
        }
    )

    errors = check_theme_completeness(tree)

    assert len(errors) == 1
    assert errors[0].rule == ValidationRule.MISSING_TOKEN
    assert errors[0].path == "text.primary"
    assert errors[0].source_file == Path("themes/light.json")


def test_check_theme_completeness_passes_identical_sets() -> None:
    tree = _tree(
        themes={
            "dark": {"bg.primary": _token("bg.primary", "#000")},
            "light": {"bg.primary": _token("bg.primary", "#fff")},
        }
    )

    assert check_theme_completeness(tree) == []


def test_check_theme_completeness_single_theme_is_trivially_complete() -> None:
    tree = _tree(themes={"dark": {"bg.primary": _token("bg.primary", "#000")}})

    assert check_theme_completeness(tree) == []


# --- check_theme_metadata --------------------------------------------------------


def test_check_theme_metadata_flags_missing_fields() -> None:
    tree = _tree(
        themes={"dark": {"bg.primary": _token("bg.primary", "#000", source_file=Path("d.json"))}},
        theme_metadata={"dark": {}},
    )

    errors = check_theme_metadata(tree)

    assert {error.rule for error in errors} == {ValidationRule.INVALID_THEME_METADATA}
    # mode, id, label, family all missing.
    assert len(errors) == 4


def test_check_theme_metadata_flags_invalid_mode_value() -> None:
    tree = _tree(
        themes={"dark": {"bg.primary": _token("bg.primary", "#000")}},
        theme_metadata={
            "dark": {"mode": "sepia", "id": "dark", "label": "Dark", "family": "osprey"}
        },
    )

    errors = check_theme_metadata(tree)

    assert len(errors) == 1
    assert "mode" in errors[0].message


def test_check_theme_metadata_accepts_well_formed_metadata() -> None:
    tree = _tree(
        themes={"dark": {"bg.primary": _token("bg.primary", "#000")}},
        theme_metadata={
            "dark": {"mode": "dark", "id": "dark", "label": "Dark", "family": "osprey"}
        },
    )

    assert check_theme_metadata(tree) == []


# --- check_theme_metadata: $extensions.family (theme "family" grouping) ---------


def test_check_theme_metadata_flags_missing_family_field() -> None:
    tree = _tree(
        themes={"dark": {"bg.primary": _token("bg.primary", "#000", source_file=Path("d.json"))}},
        theme_metadata={"dark": {"mode": "dark", "id": "dark", "label": "Dark"}},
    )

    errors = check_theme_metadata(tree)

    assert len(errors) == 1
    assert errors[0].rule == ValidationRule.INVALID_THEME_METADATA
    assert "family" in errors[0].message


def test_check_theme_metadata_flags_blank_family_value() -> None:
    tree = _tree(
        themes={"dark": {"bg.primary": _token("bg.primary", "#000")}},
        theme_metadata={"dark": {"mode": "dark", "id": "dark", "label": "Dark", "family": ""}},
    )

    errors = check_theme_metadata(tree)

    assert len(errors) == 1
    assert "family" in errors[0].message


def test_check_theme_metadata_flags_non_string_family_value() -> None:
    tree = _tree(
        themes={"dark": {"bg.primary": _token("bg.primary", "#000")}},
        theme_metadata={"dark": {"mode": "dark", "id": "dark", "label": "Dark", "family": 42}},
    )

    errors = check_theme_metadata(tree)

    assert len(errors) == 1
    assert "family" in errors[0].message


def test_check_theme_metadata_accepts_well_formed_family() -> None:
    tree = _tree(
        themes={"dark": {"bg.primary": _token("bg.primary", "#000")}},
        theme_metadata={
            "dark": {"mode": "dark", "id": "dark", "label": "Dark", "family": "high-contrast"}
        },
    )

    assert check_theme_metadata(tree) == []


# --- check_interface_mode_completeness --------------------------------------------


def test_check_interface_mode_completeness_flags_missing_mode_group() -> None:
    tree = _tree(
        themes={
            "dark": {"bg.primary": _token("x", "#000")},
            "light": {"bg.primary": _token("x", "#fff")},
        },
        interfaces={
            "demo": {
                "dark.wt-crt.opacity": _token(
                    "dark.wt-crt.opacity", "1", type_="number", source_file=Path("i/demo.json")
                )
            }
        },
    )

    errors = check_interface_mode_completeness(tree)

    missing_mode_errors = [e for e in errors if e.rule == ValidationRule.MISSING_MODE_GROUP]
    assert any(e.path == "light" for e in missing_mode_errors)


def test_check_interface_mode_completeness_flags_unexpected_mode_group() -> None:
    tree = _tree(
        themes={"dark": {"bg.primary": _token("x", "#000")}},
        interfaces={"demo": {"sepia.wt-crt.opacity": _token("p", "1", type_="number")}},
    )

    errors = check_interface_mode_completeness(tree)

    assert any(e.rule == ValidationRule.MISSING_MODE_GROUP and e.path == "sepia" for e in errors)


def test_check_interface_mode_completeness_flags_mismatched_tokens_within_modes() -> None:
    tree = _tree(
        themes={
            "dark": {"bg.primary": _token("x", "#000")},
            "light": {"bg.primary": _token("x", "#fff")},
        },
        interfaces={
            "demo": {
                "dark.wt-crt.a": _token("p", "1", type_="number", source_file=Path("i/demo.json")),
                "dark.wt-crt.b": _token("p", "1", type_="number", source_file=Path("i/demo.json")),
                "light.wt-crt.a": _token("p", "1", type_="number", source_file=Path("i/demo.json")),
            }
        },
    )

    errors = check_interface_mode_completeness(tree)

    assert len(errors) == 1
    assert errors[0].rule == ValidationRule.MISSING_TOKEN
    assert errors[0].path == "wt-crt.b"


def test_check_interface_mode_completeness_passes_matched_modes() -> None:
    tree = _tree(
        themes={
            "dark": {"bg.primary": _token("x", "#000")},
            "light": {"bg.primary": _token("x", "#fff")},
        },
        interfaces={
            "demo": {
                "dark.wt-crt.a": _token("p", "1", type_="number"),
                "light.wt-crt.a": _token("p", "1", type_="number"),
            }
        },
    )

    assert check_interface_mode_completeness(tree) == []


# --- check_interface_mode_completeness: $extensions.inherits opt-out -----------


def test_check_interface_mode_completeness_interface_inherit_opts_out_a_stem() -> None:
    # 'demo' authors real dark/light groups but opts high-contrast-dark out,
    # declaring it inherits (borrows) the 'dark' group instead of requiring
    # a duplicate, decorative-only group to be authored.
    tree = _tree(
        themes={
            "dark": {"bg.primary": _token("x", "#000")},
            "light": {"bg.primary": _token("x", "#fff")},
            "high-contrast-dark": {"bg.primary": _token("x", "#000")},
        },
        interfaces={
            "demo": {
                "dark.wt-crt.a": _token("p", "1", type_="number", source_file=Path("i/demo.json")),
                "light.wt-crt.a": _token("p", "1", type_="number", source_file=Path("i/demo.json")),
            }
        },
        interface_metadata={"demo": {"inherits": {"high-contrast-dark": "dark"}}},
    )

    errors = check_interface_mode_completeness(tree)

    assert errors == []


def test_check_interface_mode_completeness_interface_inherit_still_errors_for_undeclared_stem() -> (
    None
):
    # A stem that is neither authored nor opted-out via $extensions.inherits
    # is still a hard error -- opting out one stem must not silently excuse
    # any other missing stem.
    tree = _tree(
        themes={
            "dark": {"bg.primary": _token("x", "#000")},
            "light": {"bg.primary": _token("x", "#fff")},
            "high-contrast-dark": {"bg.primary": _token("x", "#000")},
            "high-contrast-light": {"bg.primary": _token("x", "#fff")},
        },
        interfaces={
            "demo": {
                "dark.wt-crt.a": _token("p", "1", type_="number", source_file=Path("i/demo.json")),
                "light.wt-crt.a": _token("p", "1", type_="number", source_file=Path("i/demo.json")),
            }
        },
        interface_metadata={"demo": {"inherits": {"high-contrast-dark": "dark"}}},
    )

    errors = check_interface_mode_completeness(tree)

    missing_mode_errors = [e for e in errors if e.rule == ValidationRule.MISSING_MODE_GROUP]
    # high-contrast-dark is opted out and must not error; high-contrast-light
    # was neither authored nor opted out, so it must still error.
    assert {e.path for e in missing_mode_errors} == {"high-contrast-light"}


def test_check_interface_mode_completeness_interface_inherit_rejects_dangling_base() -> None:
    # Fail-closed: an $extensions.inherits entry pointing at a base mode
    # this document does not itself define a group for is an error, not a
    # silent no-op -- otherwise a typo'd base would quietly excuse a real
    # gap.
    tree = _tree(
        themes={
            "dark": {"bg.primary": _token("x", "#000")},
            "high-contrast-dark": {"bg.primary": _token("x", "#000")},
        },
        interfaces={
            "demo": {
                "dark.wt-crt.a": _token("p", "1", type_="number", source_file=Path("i/demo.json")),
            }
        },
        interface_metadata={"demo": {"inherits": {"high-contrast-dark": "dark-typo"}}},
    )

    errors = check_interface_mode_completeness(tree)

    assert any(
        e.rule == ValidationRule.MISSING_MODE_GROUP and e.path == "high-contrast-dark"
        for e in errors
    )


# --- check_namespace_collisions ---------------------------------------------------


def test_check_namespace_collisions_flags_colliding_root() -> None:
    tree = _tree(
        themes={"dark": {"bg.primary": _token("bg.primary", "#000")}},
        interfaces={
            "demo": {
                "dark.bg.evil": _token("dark.bg.evil", "#111", source_file=Path("i/demo.json"))
            }
        },
    )

    errors = check_namespace_collisions(tree)

    assert len(errors) == 1
    assert errors[0].rule == ValidationRule.NAMESPACE_COLLISION
    assert "bg" in errors[0].message


def test_check_namespace_collisions_allows_distinct_namespace() -> None:
    tree = _tree(
        themes={"dark": {"bg.primary": _token("bg.primary", "#000")}},
        interfaces={"demo": {"dark.wt-crt.opacity": _token("p", "1", type_="number")}},
    )

    assert check_namespace_collisions(tree) == []


# --- check_promoted_primitive_collisions -------------------------------------------


def test_check_promoted_primitive_collisions_flags_theme_token_collision() -> None:
    # "text.base" is a promoted primitive scale step (--text-base); a theme
    # authoring a semantic token at the same dot-path would collide on the
    # same emitted CSS custom property name.
    tree = _tree(
        primitives={"text.base": _token("text.base", "11px", type_="dimension")},
        themes={"dark": {"text.base": _token("text.base", "#000000")}},
    )

    errors = check_promoted_primitive_collisions(tree)

    assert len(errors) == 1
    assert errors[0].rule == ValidationRule.PROMOTED_PRIMITIVE_COLLISION
    assert "text.base" in errors[0].message


def test_check_promoted_primitive_collisions_flags_interface_token_collision() -> None:
    tree = _tree(
        primitives={"space.1": _token("space.1", "4px", type_="dimension")},
        interfaces={
            "demo": {
                "dark.space.1": _token(
                    "dark.space.1", "8px", type_="dimension", source_file=Path("i/demo.json")
                )
            }
        },
    )

    errors = check_promoted_primitive_collisions(tree)

    assert len(errors) == 1
    assert errors[0].rule == ValidationRule.PROMOTED_PRIMITIVE_COLLISION
    assert "space.1" in errors[0].message


def test_check_promoted_primitive_collisions_allows_non_colliding_tokens() -> None:
    tree = _tree(
        primitives={"space.1": _token("space.1", "4px", type_="dimension")},
        themes={"dark": {"bg.primary": _token("bg.primary", "#000000")}},
        interfaces={"demo": {"dark.wt-crt.opacity": _token("p", "1", type_="number")}},
    )

    assert check_promoted_primitive_collisions(tree) == []


# --- check_wcag_gates --------------------------------------------------------------


def _wcag_theme(*, text_primary: str, bg_primary: str) -> dict[str, ResolvedToken]:
    return {
        "text.primary": _token("text.primary", text_primary, source_file=Path("themes/dark.json")),
        "text.secondary": _token(
            "text.secondary", text_primary, source_file=Path("themes/dark.json")
        ),
        "text.muted": _token("text.muted", text_primary, source_file=Path("themes/dark.json")),
        "accent.base": _token("accent.base", text_primary, source_file=Path("themes/dark.json")),
        "bg.primary": _token("bg.primary", bg_primary, source_file=Path("themes/dark.json")),
    }


def test_check_wcag_gates_passes_high_contrast_pair() -> None:
    tree = _tree(themes={"dark": _wcag_theme(text_primary="#ffffff", bg_primary="#000000")})

    assert check_wcag_gates(tree) == []


def test_check_wcag_gates_flags_low_contrast_pair() -> None:
    tree = _tree(themes={"dark": _wcag_theme(text_primary="#050505", bg_primary="#000000")})

    errors = check_wcag_gates(tree)

    assert len(errors) == 4  # all four gates share the same failing pair here
    assert {error.rule for error in errors} == {ValidationRule.WCAG_CONTRAST}


def test_check_wcag_gates_skips_gate_when_token_missing() -> None:
    tree = _tree(themes={"dark": {"bg.primary": _token("bg.primary", "#000000")}})

    assert check_wcag_gates(tree) == []


def test_wcag_gates_constant_matches_proposal_pairs_and_thresholds() -> None:
    # WCAG_GATES is the AA (default/"osprey" family) tuple. See
    # test_wcag_gates_aaa_constant_matches_high_contrast_thresholds below for
    # the AAA ("high-contrast" family) tuple.
    pairs = {(gate.foreground, gate.background, gate.minimum) for gate in WCAG_GATES}
    assert pairs == {
        ("text.primary", "bg.primary", 4.5),
        ("text.secondary", "bg.primary", 4.5),
        ("text.muted", "bg.primary", 3.0),
        ("accent.base", "bg.primary", 3.0),
        # accent.on is gated against accent.base (its fill), not bg.primary.
        ("accent.on", "accent.base", 4.5),
    }


def test_wcag_gates_aaa_constant_matches_high_contrast_thresholds() -> None:
    # AAA body text (text.primary/secondary) requires 7:1; AAA large-scale
    # text and non-text UI (text.muted/accent.base) requires 4.5:1 -- see
    # WCAG_GATES_AAA's docstring/comment for the citation.
    pairs = {(gate.foreground, gate.background, gate.minimum) for gate in WCAG_GATES_AAA}
    assert pairs == {
        ("text.primary", "bg.primary", 7.0),
        ("text.secondary", "bg.primary", 7.0),
        ("text.muted", "bg.primary", 4.5),
        ("accent.base", "bg.primary", 4.5),
        ("accent.on", "accent.base", 7.0),
    }


# --- gates_for_family: per-family WCAG gate selection ---------------------------


def test_gates_for_family_selects_aaa_tuple_for_high_contrast() -> None:
    assert gates_for_family("high-contrast") == WCAG_GATES_AAA


def test_gates_for_family_selects_aa_tuple_for_osprey() -> None:
    assert gates_for_family("osprey") == WCAG_GATES


def test_gates_for_family_falls_back_to_aa_for_unknown_family() -> None:
    # Fail-closed: an unrecognized or unspecified family never silently
    # gets a looser bar than the AA default.
    assert gates_for_family("some-unknown-family") == WCAG_GATES
    assert gates_for_family(None) == WCAG_GATES


def test_check_wcag_gates_applies_aaa_minimums_for_high_contrast_family() -> None:
    # #767676 vs #ffffff is ~4.54:1: clears AA (4.5/3.0) on every gate, but
    # only clears the AAA large-text/non-text floor (4.5) -- not the AAA
    # body-text floor (7.0). So under family='high-contrast' exactly the
    # text.primary/text.secondary gates must fail.
    tree = _tree(
        themes={"dark": _wcag_theme(text_primary="#767676", bg_primary="#ffffff")},
        theme_metadata={
            "dark": {"mode": "dark", "id": "dark", "label": "Dark", "family": "high-contrast"}
        },
    )

    errors = check_wcag_gates(tree)

    assert {error.path for error in errors} == {"text.primary", "text.secondary"}
    assert all(error.rule == ValidationRule.WCAG_CONTRAST for error in errors)


def test_check_wcag_gates_applies_aa_minimums_for_osprey_family() -> None:
    # Same ~4.54:1 pair clears every AA gate.
    tree = _tree(
        themes={"dark": _wcag_theme(text_primary="#767676", bg_primary="#ffffff")},
        theme_metadata={
            "dark": {"mode": "dark", "id": "dark", "label": "Dark", "family": "osprey"}
        },
    )

    assert check_wcag_gates(tree) == []


def test_check_wcag_gates_falls_back_to_aa_for_unspecified_family() -> None:
    # No theme_metadata entry at all for 'dark' -- must not silently pick a
    # looser bar than AA.
    tree = _tree(themes={"dark": _wcag_theme(text_primary="#767676", bg_primary="#ffffff")})

    assert check_wcag_gates(tree) == []


# --- validate_token_tree / assert_valid --------------------------------------------


def test_validate_token_tree_reports_every_failure_not_just_first() -> None:
    tree = _tree(
        themes={
            "dark": {
                "bg.primary": _token("bg.primary", "not-a-color"),
                "accent.base": _token(
                    "accent.base",
                    "{missing}",
                    alias_status=AliasStatus.DANGLING,
                    alias_target="missing",
                ),
            },
            "light": {"bg.primary": _token("bg.primary", "#ffffff")},
        }
    )

    errors = validate_token_tree(tree)
    rules = {error.rule for error in errors}

    # At minimum: invalid color, dangling alias, and theme-completeness
    # (dark has accent.base light doesn't; light is missing nothing dark
    # has... just assert several distinct rules fired together).
    assert ValidationRule.INVALID_COLOR in rules
    assert ValidationRule.DANGLING_ALIAS in rules
    assert ValidationRule.MISSING_TOKEN in rules
    assert len(errors) > 2


def test_validate_token_tree_empty_tree_is_valid() -> None:
    assert validate_token_tree(_tree()) == []


def test_assert_valid_raises_with_all_errors() -> None:
    tree = _tree(themes={"dark": {"bg.primary": _token("bg.primary", "not-a-color")}})

    with pytest.raises(TokenValidationError) as excinfo:
        assert_valid(tree)

    assert len(excinfo.value.errors) == 1
    assert excinfo.value.errors[0].rule == ValidationRule.INVALID_COLOR


def test_assert_valid_does_not_raise_for_clean_tree() -> None:
    assert_valid(_tree())  # no exception


def test_validation_error_str_includes_path_and_file() -> None:
    tree = _tree(
        themes={"dark": {"bg.primary": _token("bg.primary", "nope", source_file=Path("d.json"))}}
    )

    (error,) = check_color_syntax(tree)

    assert str(error) == f"d.json (bg.primary): {error.message}"


# --- Full pipeline: a tiny, valid tokens/ tree on disk -----------------------------


def test_full_pipeline_clean_tree_validates_with_zero_errors(tmp_path: Path) -> None:
    (tmp_path / "themes").mkdir()
    (tmp_path / "interfaces").mkdir()

    (tmp_path / "core.json").write_text(
        json.dumps(
            {
                "color": {
                    "teal": {"500": {"$value": "#14b8a6", "$type": "color"}},
                    "slate": {
                        "50": {"$value": "#f8fafc", "$type": "color"},
                        "900": {"$value": "#0a0f1a", "$type": "color"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    dark = {
        "$extensions": {"id": "dark", "label": "Dark", "mode": "dark", "family": "osprey"},
        "bg": {"primary": {"$value": "{color.slate.900}", "$type": "color"}},
        "text": {"primary": {"$value": "#ffffff", "$type": "color"}},
        "accent": {"base": {"$value": "{color.teal.500}", "$type": "color"}},
        "terminal": {"cursor": {"$value": "{color.teal.500}", "$type": "color"}},
    }
    light = {
        "$extensions": {"id": "light", "label": "Light", "mode": "light", "family": "osprey"},
        "bg": {"primary": {"$value": "{color.slate.50}", "$type": "color"}},
        "text": {"primary": {"$value": "#000000", "$type": "color"}},
        # A darker teal step than dark's accent.base — the light theme
        # needs enough contrast against its near-white background to
        # clear the 3:1 non-text WCAG gate (mirrors how real tokens pick
        # a different ramp step per theme for the same semantic role).
        "accent": {"base": {"$value": "#065f5c", "$type": "color"}},
        "terminal": {"cursor": {"$value": "{color.teal.500}", "$type": "color"}},
    }
    (tmp_path / "themes" / "dark.json").write_text(json.dumps(dark), encoding="utf-8")
    (tmp_path / "themes" / "light.json").write_text(json.dumps(light), encoding="utf-8")

    demo = {
        "dark": {"wt-crt": {"opacity": {"$value": "1", "$type": "number"}}},
        "light": {"wt-crt": {"opacity": {"$value": "0", "$type": "number"}}},
    }
    (tmp_path / "interfaces" / "demo.json").write_text(json.dumps(demo), encoding="utf-8")

    tree = load_token_tree(tmp_path)

    assert validate_token_tree(tree) == []


# --- Regression: the real, already-committed tokens/ tree -------------------------


@pytest.mark.skipif(not REAL_TOKENS_DIR.is_dir(), reason="real tokens/ tree not present yet")
def test_real_tokens_tree_currently_validates_clean() -> None:
    tree = load_token_tree(REAL_TOKENS_DIR)

    errors = validate_token_tree(tree)

    assert errors == [], "\n".join(str(error) for error in errors)
