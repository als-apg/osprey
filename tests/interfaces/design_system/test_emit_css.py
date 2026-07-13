"""Tests for osprey.interfaces.design_system.generator.emit_css.

Builds small TokenTree instances directly in Python (this task's file
ownership doesn't include fixtures/**) plus one regression test against
the real, already-committed tokens/ tree.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

import osprey.interfaces.design_system as design_system_pkg
from osprey.interfaces.design_system.generator.emit_css import css_variable_name, emit_css
from osprey.interfaces.design_system.generator.model import (
    AliasStatus,
    ResolvedToken,
    TokenTree,
    load_token_tree,
)

REAL_TOKENS_DIR = Path(design_system_pkg.__file__).parent / "tokens"


def _token(
    path: str,
    value: object,
    *,
    type_: str | None = "color",
    alias_status: AliasStatus = AliasStatus.NOT_ALIAS,
    source_file: Path = Path("<test>"),
) -> ResolvedToken:
    return ResolvedToken(
        path=path,
        value=value,
        type=type_,
        description=None,
        extensions={},
        source_file=source_file,
        alias_status=alias_status,
        alias_target=None,
    )


def _small_tree() -> TokenTree:
    """A hand-built, minimal-but-representative two-theme tree."""
    primitives = {
        "font.display": _token("font.display", "'Outfit', sans-serif", type_="fontFamily"),
        "font.mono": _token("font.mono", "'JetBrains Mono', monospace", type_="fontFamily"),
        "color.teal.300": _token("color.teal.300", "#319795"),
    }

    def theme_tokens(bg: str, text: str, accent: str) -> dict[str, ResolvedToken]:
        return {
            "bg.primary": _token("bg.primary", bg),
            "text.primary": _token("text.primary", text),
            "accent.base": _token("accent.base", accent),
            "amber.hover": _token("amber.hover", "#e08e00"),
            "status.error-hover": _token("status.error-hover", "#b91c1c"),
            "tint.neutral.20": _token("tint.neutral.20", "rgba(148, 163, 184, 0.20)"),
            "tint.accent.04": _token("tint.accent.04", "rgba(79, 209, 197, 0.04)"),
            "terminal.cursor": _token("terminal.cursor", "#4fd1c5"),
            "terminal.ansi.bright-cyan": _token("terminal.ansi.bright-cyan", "#67e8f9"),
            "chart.paper-bg": _token("chart.paper-bg", bg),
            "chart.series.1": _token("chart.series.1", "#4fd1c5"),
            "code.theme": _token("code.theme", "highlight.js atom-one-dark theme", type_="string"),
        }

    themes = {
        "dark": theme_tokens("#0a0f1a", "#f1f5f9", "#319795"),
        "light": theme_tokens("#f7f9fc", "#0c1322", "#0a8f8c"),
    }
    theme_metadata = {
        "dark": {"id": "dark", "label": "Dark", "mode": "dark"},
        "light": {"id": "light", "label": "Light", "mode": "light"},
    }

    interfaces = {
        "web_terminal": {
            "dark.wt-crt.opacity": _token(
                "dark.wt-crt.opacity", "1", type_="number", source_file=Path("wt.json")
            ),
            "light.wt-crt.opacity": _token(
                "light.wt-crt.opacity", "0", type_="number", source_file=Path("wt.json")
            ),
        },
        "channel_finder": {
            "dark.cf-font-pv": _token(
                "dark.cf-font-pv",
                "'Syne Mono', monospace",
                type_="fontFamily",
                source_file=Path("cf.json"),
            ),
            "light.cf-font-pv": _token(
                "light.cf-font-pv",
                "'Syne Mono', monospace",
                type_="fontFamily",
                source_file=Path("cf.json"),
            ),
        },
    }

    return TokenTree(
        primitives=primitives,
        themes=themes,
        interfaces=interfaces,
        theme_metadata=theme_metadata,
        interface_metadata={"web_terminal": {}, "channel_finder": {}},
    )


# --- css_variable_name -----------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("accent.base", "--color-accent"),
        ("accent.light", "--color-accent-light"),
        ("amber.base", "--color-amber"),
        ("amber.light", "--color-amber-light"),
        ("amber.hover", "--color-amber-hover"),
        ("status.success", "--color-success"),
        ("status.warning", "--color-warning"),
        ("status.error", "--color-error"),
        ("status.error-hover", "--color-error-hover"),
        ("tint.neutral.20", "--neutral-tint-20"),
        ("tint.accent.04", "--accent-tint-04"),
        ("terminal.ansi.bright-cyan", "--ansi-bright-cyan"),
        ("terminal.ansi.background", "--ansi-background"),
        ("terminal.cursor", "--terminal-cursor"),
        ("terminal.text", "--terminal-text"),
        ("terminal.selection", "--terminal-selection"),
        ("bg.primary", "--bg-primary"),
        ("text.muted", "--text-muted"),
        ("border.default", "--border-default"),
        ("overlay.black-20", "--overlay-black-20"),
        ("diff.word-del-bg", "--diff-word-del-bg"),
        ("chart.paper-bg", "--chart-paper-bg"),
        ("chart.series.1", "--chart-series-1"),
        ("wt-crt.scanline-opacity", "--wt-crt-scanline-opacity"),
        ("art-violet.base", "--art-violet-base"),
        ("ariel-score.high", "--ariel-score-high"),
        ("lat-led.ready", "--lat-led-ready"),
        ("cf-font-pv", "--cf-font-pv"),
    ],
)
def test_css_variable_name_mapping(path: str, expected: str) -> None:
    assert css_variable_name(path) == expected


def test_css_variable_name_excludes_code_group() -> None:
    assert css_variable_name("code.theme") is None
    assert css_variable_name("code") is None


# --- emit_css structure -----------------------------------------------------------


def test_emit_css_header_has_no_edit_warning_and_no_timestamp() -> None:
    css = emit_css(_small_tree())

    assert css.startswith("/* AUTO-GENERATED")
    assert "DO NOT EDIT" in css
    # No ISO-ish date/time patterns anywhere in the header/file.
    assert not re.search(r"\d{4}-\d{2}-\d{2}", css)
    assert not re.search(r"\d{2}:\d{2}:\d{2}", css)


def test_emit_css_default_dark_theme_combined_with_root() -> None:
    css = emit_css(_small_tree())

    assert ':root, [data-theme="dark"] {' in css
    assert "color-scheme: dark;" in css


def test_emit_css_other_theme_is_its_own_block() -> None:
    css = emit_css(_small_tree())

    assert '[data-theme="light"] {' in css
    # Light block must not be combined with :root.
    assert ':root, [data-theme="light"]' not in css


def test_emit_css_fonts_emitted_once_in_default_block_only() -> None:
    css = emit_css(_small_tree())

    assert css.count("--font-display:") == 1
    assert css.count("--font-mono:") == 1

    dark_block, light_block = _extract_blocks(css)
    assert "--font-display" in dark_block
    assert "--font-display" not in light_block


def test_emit_css_excludes_code_group_entirely() -> None:
    css = emit_css(_small_tree())

    assert "--code-theme" not in css
    assert "code.theme" not in css


def test_emit_css_extension_tokens_use_theme_specific_value() -> None:
    css = emit_css(_small_tree())
    dark_block, light_block = _extract_blocks(css)

    assert "--wt-crt-opacity: 1;" in dark_block
    assert "--wt-crt-opacity: 0;" in light_block
    assert "--cf-font-pv: 'Syne Mono', monospace;" in dark_block
    assert "--cf-font-pv: 'Syne Mono', monospace;" in light_block


def test_emit_css_semantic_values_differ_per_theme() -> None:
    css = emit_css(_small_tree())
    dark_block, light_block = _extract_blocks(css)

    assert "--bg-primary: #0a0f1a;" in dark_block
    assert "--bg-primary: #f7f9fc;" in light_block
    assert "--color-accent: #319795;" in dark_block
    assert "--color-accent: #0a8f8c;" in light_block


def test_emit_css_source_order_preserved_within_block() -> None:
    css = emit_css(_small_tree())
    dark_block, _ = _extract_blocks(css)

    # bg.primary is declared before text.primary in the source dict.
    assert dark_block.index("--bg-primary") < dark_block.index("--text-primary")
    assert dark_block.index("--text-primary") < dark_block.index("--color-accent")


def test_emit_css_is_deterministic() -> None:
    tree = _small_tree()
    assert emit_css(tree) == emit_css(tree)


def test_emit_css_raises_on_unresolved_alias() -> None:
    tree = _small_tree()
    tree.themes["dark"]["bg.primary"] = _token(
        "bg.primary", "{missing}", alias_status=AliasStatus.DANGLING
    )

    with pytest.raises(ValueError, match="unresolved alias"):
        emit_css(tree)


def test_emit_css_raises_when_no_dark_theme() -> None:
    tree = dataclasses.replace(
        _small_tree(),
        theme_metadata={
            "dark": {"id": "dark", "label": "Dark", "mode": "light"},
            "light": {"id": "light", "label": "Light", "mode": "light"},
        },
    )

    with pytest.raises(ValueError, match="no theme declares"):
        emit_css(tree)


# --- Hook-cleanliness / idempotence (trailing-whitespace, end-of-file-fixer) ------


def test_emit_css_has_no_trailing_whitespace_on_any_line() -> None:
    css = emit_css(_small_tree())

    for line in css.splitlines():
        assert line == line.rstrip(), f"trailing whitespace on line: {line!r}"


def test_emit_css_has_exactly_one_trailing_newline() -> None:
    css = emit_css(_small_tree())

    assert css == css.rstrip("\n") + "\n"


def test_emit_css_is_idempotent_under_whitespace_hooks() -> None:
    css = emit_css(_small_tree())

    # Simulates pre-commit's trailing-whitespace + end-of-file-fixer hooks;
    # the generated file must already be a fixed point of both.
    fixed = "\n".join(line.rstrip() for line in css.splitlines()) + "\n"
    assert fixed == css


# --- Regression: the real, already-committed tokens/ tree -------------------------


@pytest.mark.skipif(not REAL_TOKENS_DIR.is_dir(), reason="real tokens/ tree not present yet")
def test_real_tokens_tree_emits_required_hard_constraint_names() -> None:
    css = emit_css(load_token_tree(REAL_TOKENS_DIR))

    for required in (
        "--bg-primary:",
        "--text-primary:",
        "--font-display:",
        "--neutral-tint-20:",
        "--neutral-tint-35:",
    ):
        assert required in css, f"missing required CSS variable: {required}"


@pytest.mark.skipif(not REAL_TOKENS_DIR.is_dir(), reason="real tokens/ tree not present yet")
def test_real_tokens_tree_css_is_hook_clean_and_deterministic() -> None:
    tree = load_token_tree(REAL_TOKENS_DIR)
    css = emit_css(tree)

    assert css == emit_css(tree)
    assert css == css.rstrip("\n") + "\n"
    for line in css.splitlines():
        assert line == line.rstrip()


def _extract_blocks(css: str) -> tuple[str, str]:
    """Split emitted CSS into (dark_block, light_block) bodies for assertions."""
    dark_start = css.index(':root, [data-theme="dark"]')
    light_start = css.index('[data-theme="light"]')
    return css[dark_start:light_start], css[light_start:]


def _blocks_by_theme(css: str) -> dict[str, str]:
    """Map each emitted theme id to the body text of its ``{ ... }`` block."""
    blocks: dict[str, str] = {}
    for match in re.finditer(r'(?::root, )?\[data-theme="([^"]+)"\][^{]*\{([^}]*)\}', css):
        blocks[match.group(1)] = match.group(2)
    return blocks


def _declared_names(body: str) -> set[str]:
    """Every ``--name`` a CSS block body declares."""
    return set(re.findall(r"(--[a-z0-9-]+)\s*:", body))


# --- $extensions.inherits: opted-out theme borrows a base mode's tokens -----------


def _inherits_tree() -> TokenTree:
    """A 4-theme tree where an interface opts the high-contrast twins out via inherits.

    ``web_terminal`` authors only ``dark``/``light`` groups and maps the two
    high-contrast stems onto them, exactly as the real interface docs do.
    """
    primitives = {
        "font.display": _token("font.display", "'Outfit', sans-serif", type_="fontFamily"),
    }

    def theme_tokens(bg: str) -> dict[str, ResolvedToken]:
        return {"bg.primary": _token("bg.primary", bg)}

    themes = {
        "dark": theme_tokens("#0a0f1a"),
        "light": theme_tokens("#f7f9fc"),
        "high-contrast-dark": theme_tokens("#000000"),
        "high-contrast-light": theme_tokens("#ffffff"),
    }

    def meta(id_: str, mode: str, family: str) -> dict[str, str]:
        return {"id": id_, "label": id_, "mode": mode, "family": family}

    theme_metadata = {
        "dark": meta("dark", "dark", "osprey"),
        "light": meta("light", "light", "osprey"),
        "high-contrast-dark": meta("high-contrast-dark", "dark", "high-contrast"),
        "high-contrast-light": meta("high-contrast-light", "light", "high-contrast"),
    }

    interfaces = {
        "web_terminal": {
            "dark.wt-crt.opacity": _token(
                "dark.wt-crt.opacity", "1", type_="number", source_file=Path("wt.json")
            ),
            "light.wt-crt.opacity": _token(
                "light.wt-crt.opacity", "0", type_="number", source_file=Path("wt.json")
            ),
        }
    }

    return TokenTree(
        primitives=primitives,
        themes=themes,
        interfaces=interfaces,
        theme_metadata=theme_metadata,
        interface_metadata={
            "web_terminal": {
                "inherits": {"high-contrast-dark": "dark", "high-contrast-light": "light"}
            }
        },
    )


def test_emit_css_inherited_interface_tokens_reach_opted_out_themes() -> None:
    """An interface that opts a theme out via inherits still emits into that theme's block.

    Regression: emit_css used to match interface tokens only by an exact
    ``{stem}.`` prefix, so an interface authoring only ``dark``/``light`` and
    mapping the high-contrast twins onto them contributed *nothing* to either
    high-contrast block — the tokens silently fell through to the dark ``:root``
    cascade (near-black text on the high-contrast-light page).
    """
    blocks = _blocks_by_theme(emit_css(_inherits_tree()))

    # high-contrast-dark borrows dark's group; high-contrast-light borrows light's.
    assert "--wt-crt-opacity: 1;" in blocks["high-contrast-dark"]
    assert "--wt-crt-opacity: 0;" in blocks["high-contrast-light"]


@pytest.mark.skipif(not REAL_TOKENS_DIR.is_dir(), reason="real tokens/ tree not present yet")
def test_real_tokens_every_theme_block_defines_every_interface_token() -> None:
    """No interface token may appear in one theme block but be missing from another.

    The union of interface extension token names (mode-stripped) must be
    present in *every* emitted theme block — the invariant the
    ``$extensions.inherits`` resolution guarantees. Without it, the
    high-contrast themes emit none of the opted-out interfaces' tokens and
    inherit the dark values instead.
    """
    tree = load_token_tree(REAL_TOKENS_DIR)
    blocks = _blocks_by_theme(emit_css(tree))

    expected: set[str] = set()
    for tokens in tree.interfaces.values():
        for path in tokens:
            _mode, _, rest = path.partition(".")
            name = css_variable_name(rest)
            if name is not None:
                expected.add(name)

    for theme_id, body in blocks.items():
        missing = expected - _declared_names(body)
        assert not missing, f"theme {theme_id!r} block missing interface tokens: {sorted(missing)}"
