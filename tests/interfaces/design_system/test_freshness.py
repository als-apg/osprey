"""The freshness gate: the real, committed generated artifacts must never drift.

Unlike every other test in this package, this one is deliberately NOT
hermetic — it is the drift gate CI (and ``build --check``) rely on, so it
must exercise the real ``tokens/`` source tree against the real,
committed ``static/css/tokens.css``, ``static/js/tokens.js``, and
``static/js/theme-boot.js``. If a token source changes without
regenerating these three files, this test (and the equivalent
``build --check`` invocation) fails.

This module also carries the two other things Task 1.7 is responsible
for finalizing, both checked against the real token tree:

- the CSS custom property names ``base.css`` (hand-written, a
  different task's file) hard-depends on existing;
- the rename-mapping table from every name the web-terminal
  ``variables.css``/``theme-light.css`` vocabulary defines to whatever
  the generator actually emits for it — verbatim, or a documented
  rename. Downstream migration tasks (2.2 onward) delete those two files
  and need this mapping to know what to point every ``var(--old-name)``
  call site at. See the inline ``_DOCUMENTED_RENAMES`` table below for the
  four names Task 1.1 originally missed (surfaced only once the real
  build ran and the real CSS files were diffed against it) and the
  ``--crt-*``/``--welcome-*`` -> ``--wt-crt-*``/``--wt-welcome-*``
  namespacing already planned in Task 1.1.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import osprey.interfaces.design_system as design_system_pkg
from osprey.integrations.canopy import HOST_CONTRACT_NAMES
from osprey.interfaces.design_system.generator.build import (
    DEFAULT_STATIC_DIR,
    DEFAULT_TOKENS_DIR,
    build_artifacts,
    check_artifacts,
)
from osprey.interfaces.design_system.generator.emit_css import emit_css
from osprey.interfaces.design_system.generator.model import load_token_tree
from osprey.interfaces.design_system.generator.validate import (
    contrast_ratio,
    gates_for_family,
    parse_color,
)

_VARIABLES_CSS = (
    Path(design_system_pkg.__file__).parents[1]
    / "web_terminal"
    / "static"
    / "css"
    / "variables.css"
)
_THEME_LIGHT_CSS = (
    Path(design_system_pkg.__file__).parents[1]
    / "web_terminal"
    / "static"
    / "css"
    / "theme-light.css"
)

#: Every name the OLD web-terminal ``variables.css``/``theme-light.css``
#: vocabulary defines that the generator does NOT emit verbatim, mapped to
#: the name it emits instead. Two groups:
#:
#: 1. ``--crt-*``/``--welcome-*`` -> ``--wt-crt-*``/``--wt-welcome-*``:
#:    planned in Task 1.1 (these were always going to be namespaced
#:    extension tokens, not semantic ones).
#: 2. ``--transition-fast``, ``--transition-normal``,
#:    ``--accent-system-tint-04``: Task 1.1
#:    genuinely missed these four theme-independent layout/motion values
#:    and one color tint — surfaced only by running the real build and
#:    diffing its output against the real, hand-authored CSS files (this
#:    task's whole purpose). All five are live dependencies of
#:    scaffold.css/operator.css/terminal.css/files.css/settings.css/
#:    drawer.css/sessions.css/md-rendered.css/session.html (60+ call
#:    sites) and were added to ``tokens/interfaces/web_terminal.json``
#:    as ``wt-*`` extension tokens rather than left bare, to avoid
#:    reopening the flat-namespace collision risk the wt-/art-/ariel-/
#:    lat-/cf- convention exists to close (namespace-collision is a
#:    validated invariant, see ``validate.check_namespace_collisions``).
#:    Every migration task that still references one of these five old
#:    names by the time it deletes variables.css/theme-light.css must
#:    rewrite it to the ``wt-`` name instead.
_DOCUMENTED_RENAMES: dict[str, str] = {
    "crt-scanline-opacity": "wt-crt-scanline-opacity",
    "crt-vignette-opacity": "wt-crt-vignette-opacity",
    "crt-glow-opacity": "wt-crt-glow-opacity",
    "crt-bezel-shadow": "wt-crt-bezel-shadow",
    "crt-scanline-bg": "wt-crt-scanline-bg",
    "welcome-bg": "wt-welcome-bg",
    "welcome-text": "wt-welcome-text",
    "welcome-text-glow": "wt-welcome-text-glow",
    "welcome-link": "wt-welcome-link",
    "welcome-link-hover": "wt-welcome-link-hover",
    "welcome-prompt": "wt-welcome-prompt",
    "welcome-cursor": "wt-welcome-cursor",
    "transition-fast": "wt-transition-fast",
    "transition-normal": "wt-transition-normal",
    "accent-system-tint-04": "wt-accent-system-tint-04",
}

#: CSS custom property names ``design_system/static/css/base.css``
#: (a hand-written file) references directly. A hard constraint: whatever
#: token sources/emitter changes happen, every one of these names must keep
#: existing in the emitted CSS, or the hand-written file silently renders
#: with unresolved variables.
#:
#: Keep this list equal to the file's actual ``var(--…)`` references:
#:     grep -o 'var(--[a-zA-Z0-9-]*' base.css | sed 's/var(--//' | sort -u
_BASE_CSS_HARD_DEPENDENCIES: tuple[str, ...] = (
    "bg-elevated",
    "bg-primary",
    "bg-secondary",
    "border-accent",
    "border-default",
    "color-accent",
    "color-error",
    "duration-base",
    "duration-fast",
    "duration-instant",
    "error-tint-10",
    "font-display",
    "font-mono",
    "leading-none",
    "leading-normal",
    "neutral-tint-20",
    "neutral-tint-35",
    "radius-full",
    "radius-lg",
    "radius-sm",
    "radius-xl",
    "shadow-dropdown",
    "space-1",
    "space-2",
    "space-3",
    "text-base",
    "text-lg",
    "text-md",
    "text-muted",
    "text-primary",
    "text-secondary",
    "text-sm",
    "text-xl",
    "text-xs",
    "weight-medium",
    "weight-regular",
    "weight-semibold",
    "z-dropdown",
    "z-overlay",
)

_VAR_DECLARATION_PATTERN = re.compile(r"--([a-zA-Z0-9-]+)\s*:")


def _declared_var_names(css_text: str) -> set[str]:
    """Every ``--name`` a CSS custom-property declaration defines in ``css_text``."""
    return set(_VAR_DECLARATION_PATTERN.findall(css_text))


# --- The freshness gate itself ----------------------------------------------------


def test_regeneration_is_byte_identical_to_checked_in_artifacts() -> None:
    """The actual drift gate: real tokens/ must regenerate the real static/ files exactly."""
    artifacts = build_artifacts(DEFAULT_TOKENS_DIR)

    diffs = check_artifacts(artifacts, DEFAULT_STATIC_DIR)

    assert diffs == [], "\n\n".join(
        f"{diff.relative_path} has drifted:\n{diff.unified_diff}" for diff in diffs
    )


def test_check_cli_reports_up_to_date_against_the_real_tree(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The exact `build --check` invocation the validation gate runs, in-process."""
    from osprey.interfaces.design_system.generator.build import main

    exit_code = main(["--check"])

    assert exit_code == 0
    assert "up to date" in capsys.readouterr().out


def test_check_cli_subprocess_matches_validation_gate_command() -> None:
    """Smoke test the literal `python -m ...generator.build --check` the gate runs."""
    result = subprocess.run(
        [sys.executable, "-m", "osprey.interfaces.design_system.generator.build", "--check"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "up to date" in result.stdout


def test_generated_artifacts_are_hook_clean_on_disk() -> None:
    """The checked-in files themselves, not just the in-memory render, are hook-clean."""
    for relative_path in (
        DEFAULT_STATIC_DIR / "css" / "tokens.css",
        DEFAULT_STATIC_DIR / "js" / "tokens.js",
        DEFAULT_STATIC_DIR / "js" / "theme-boot.js",
    ):
        content = relative_path.read_text(encoding="utf-8")
        assert "\r" not in content, f"{relative_path}: must use \\n line endings"
        assert content.endswith("\n") and not content.endswith("\n\n"), (
            f"{relative_path}: must have exactly one trailing newline"
        )
        for line in content.split("\n"):
            assert line == line.rstrip(), f"{relative_path}: trailing whitespace on {line!r}"


# --- Hard constraint: base.css's five CSS variable names ---------------------------


def test_base_css_hard_dependencies_are_emitted() -> None:
    tree = load_token_tree(DEFAULT_TOKENS_DIR)
    css = emit_css(tree)
    names = _declared_var_names(css)

    missing = [name for name in _BASE_CSS_HARD_DEPENDENCIES if name not in names]
    assert missing == [], f"base.css depends on --{{{', '.join(missing)}}}, not emitted"


def test_base_css_hard_dependency_list_matches_the_file() -> None:
    """The pinned list above must BE base.css's references, not a subset of them.

    Without this, the gate rots the moment someone adds a ``var(--…)`` to
    base.css without touching this file — and an unpinned token can then be
    renamed away, leaving the hand-written stylesheet quietly unresolved.
    """
    base_css = (DEFAULT_STATIC_DIR / "css" / "base.css").read_text(encoding="utf-8")
    referenced = set(re.findall(r"var\(--([a-zA-Z0-9-]+)", base_css))

    assert referenced == set(_BASE_CSS_HARD_DEPENDENCIES), (
        "_BASE_CSS_HARD_DEPENDENCIES is out of sync with base.css: "
        f"unpinned={sorted(referenced - set(_BASE_CSS_HARD_DEPENDENCIES))}, "
        f"stale={sorted(set(_BASE_CSS_HARD_DEPENDENCIES) - referenced)}"
    )


# --- The rename-mapping table: every old variables.css name is accounted for -------


@pytest.mark.skipif(
    not (_VARIABLES_CSS.is_file() and _THEME_LIGHT_CSS.is_file()),
    reason="web_terminal variables.css/theme-light.css already deleted (post-migration)",
)
def test_every_variables_css_name_is_emitted_verbatim_or_documented_renamed() -> None:
    old_names = _declared_var_names(
        _VARIABLES_CSS.read_text(encoding="utf-8")
    ) | _declared_var_names(_THEME_LIGHT_CSS.read_text(encoding="utf-8"))

    tree = load_token_tree(DEFAULT_TOKENS_DIR)
    new_names = _declared_var_names(emit_css(tree))

    unaccounted = sorted(
        name
        for name in old_names
        if name not in new_names and _DOCUMENTED_RENAMES.get(name) not in new_names
    )
    assert unaccounted == [], (
        f"old variables.css/theme-light.css name(s) {unaccounted} are neither emitted "
        "verbatim nor covered by _DOCUMENTED_RENAMES"
    )


def test_documented_renames_all_resolve_to_real_emitted_names() -> None:
    # Guards the mapping table itself against bit-rot: every declared
    # rename target must actually exist in what the generator emits today.
    tree = load_token_tree(DEFAULT_TOKENS_DIR)
    new_names = _declared_var_names(emit_css(tree))

    dangling = sorted(new for new in _DOCUMENTED_RENAMES.values() if new not in new_names)
    assert dangling == [], f"_DOCUMENTED_RENAMES target(s) {dangling} are not actually emitted"


# --- The CANOPY host adapter: canopy-host.css ---------------------------------------
#
# ``canopy-host.css`` (hand-written) republishes OSPREY's tokens under the
# CANOPY host contract's names. An embedded CANOPY page reads nothing else
# from OSPREY, so three things must hold or the page silently falls back to
# unresolved variables or unreadable pairs: every token the adapter reads
# exists in every theme, every contract name is published, and the pairs
# CANOPY paints from those names clear the same contrast gate OSPREY
# sets for the theme's own family.

_CANOPY_HOST_CSS = DEFAULT_STATIC_DIR / "css" / "canopy-host.css"
_PALETTE_CSS = DEFAULT_STATIC_DIR / "css" / "tokens.css"

#: The attribute CANOPY puts on its anchor element; every adapter selector
#: starts with it.
_HOST_ANCHOR = "[data-canopy-host-root]"
_HOST_PREFIX = "canopy-host-"
#: The contract-version marker CANOPY reads from the anchor.
_HOST_MARKER = "canopy-host-contract"

#: The eight theme ids, verbatim (the stems of ``tokens/themes/*.json``).
THEME_IDS: tuple[str, ...] = (
    "dark",
    "light",
    "desy-dark",
    "desy-light",
    "high-contrast-dark",
    "high-contrast-light",
    "retro-dark",
    "retro-light",
)

#: Themes where the adapter replaces the default accent-hover source
#: (``--color-accent-light``): in the first four that token equals the accent,
#: so hover would be invisible; in ``retro-light`` it is darker than the
#: accent under a dark on-accent label.
_ACCENT_HOVER_EXCEPTIONS: frozenset[str] = frozenset(
    {"dark", "desy-dark", "desy-light", "light", "retro-light"}
)

#: Gate role of a pair that carries text: it takes the family's
#: ``text.primary`` minimum (4.5, or 7.0 in the high-contrast family).
_TEXT_ROLE = "text.primary"
#: Gate role of a non-text pair (the toggle knob on its track): it takes the
#: family's ``accent.base`` minimum (3.0, or 4.5 in the high-contrast family).
_NON_TEXT_ROLE = "accent.base"

#: Every pair CANOPY paints from contract names, ``foreground/background``,
#: with the gate role that sets its minimum.
_CONTRAST_PAIR_ROLES: dict[str, str] = {
    "text-primary/bg-elevated": _TEXT_ROLE,
    "text-primary/secondary-mix": _TEXT_ROLE,
    "text-secondary/bg-secondary": _TEXT_ROLE,
    "on-accent/accent": _TEXT_ROLE,
    "on-accent/accent-hover": _TEXT_ROLE,
    "accent/selection": _TEXT_ROLE,
    "on-accent/text-muted": _NON_TEXT_ROLE,
    "on-accent/status-success": _NON_TEXT_ROLE,
}

#: Share of ``text-primary`` CANOPY mixes into ``bg-elevated`` for the
#: resting secondary button, the surface behind the "secondary-mix" pair.
_SECONDARY_MIX_SHARE = 0.12

#: The pairs that fall below their family minimum on OSPREY's own palette,
#: with the measured ratio. These are shortfalls of the token sources, which
#: already paint the same pairs in OSPREY's own chrome; the adapter matches
#: OSPREY rather than rerouting them. The test below requires the measured
#: failures to equal these keys exactly: a new shortfall fails it, and a
#: token fix that lifts a pair over its minimum fails it until the entry is
#: deleted. Never weaken a family gate, and never add an entry, to fit a
#: value.
KNOWN_SHORTFALLS: dict[tuple[str, str], float] = {
    ("light", "on-accent/status-success"): 2.92,
    ("light", "text-secondary/bg-secondary"): 4.31,
    ("desy-light", "accent/selection"): 3.48,
    ("retro-dark", "accent/selection"): 4.33,
    ("retro-light", "accent/selection"): 3.15,
}

#: Allowed distance between a measured shortfall and its pinned ratio.
_SHORTFALL_TOLERANCE = 0.02

_CSS_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_RULE_PATTERN = re.compile(r"([^{}]+)\{([^{}]*)\}")
_CSS_CUSTOM_PROPERTY_PATTERN = re.compile(r"--([a-zA-Z0-9-]+)\s*:\s*([^;]+);")
_THEME_SELECTOR_PATTERN = re.compile(r'\[data-theme="([a-z0-9-]+)"\]')
_VAR_REFERENCE_PATTERN = re.compile(r"var\(--([a-zA-Z0-9-]+)\)")
_COLOR_MIX_PATTERN = re.compile(r"color-mix\(in srgb,([^,]+),([^,]+)\)")
_MIX_PART_PATTERN = re.compile(r"(.+?)\s+(\d+(?:\.\d+)?)%")
_LENGTH_PATTERN = re.compile(r"\d+px")

#: The two named colors the adapter may use inside ``color-mix()``.
_NAMED_COLORS: dict[str, tuple[float, float, float, float]] = {
    "white": (255.0, 255.0, 255.0, 1.0),
    "black": (0.0, 0.0, 0.0, 1.0),
}

#: An sRGB color with unrounded 0-255 channels and a 0-1 alpha. Mixed and
#: composited colors stay in this form up to the luminance computation, so
#: no ratio depends on rounding a channel to an integer first.
_Rgba = tuple[float, float, float, float]


def _css_rules(css_text: str) -> list[tuple[list[str], dict[str, str]]]:
    """Every rule of ``css_text`` as (selector list, custom-property declarations)."""
    return [
        (
            [selector.strip() for selector in selectors.split(",")],
            {name: value.strip() for name, value in _CSS_CUSTOM_PROPERTY_PATTERN.findall(body)},
        )
        for selectors, body in _CSS_RULE_PATTERN.findall(_CSS_COMMENT_PATTERN.sub("", css_text))
    ]


def _canopy_host_rules() -> list[tuple[list[str], dict[str, str]]]:
    return _css_rules(_CANOPY_HOST_CSS.read_text(encoding="utf-8"))


def _token_blocks() -> dict[str, dict[str, str]]:
    """The committed ``tokens.css`` as theme id -> the names its block declares."""
    blocks: dict[str, dict[str, str]] = {}
    for selectors, declarations in _css_rules(_PALETTE_CSS.read_text(encoding="utf-8")):
        for selector in selectors:
            match = _THEME_SELECTOR_PATTERN.fullmatch(selector)
            if match:
                blocks[match.group(1)] = declarations
    assert set(blocks) == set(THEME_IDS), f"tokens.css theme blocks: {sorted(blocks)}"
    return blocks


def _host_base_block() -> dict[str, str]:
    """The adapter's one theme-independent block: what the bare anchor declares."""
    base_blocks = [
        declarations
        for selectors, declarations in _canopy_host_rules()
        if selectors == [_HOST_ANCHOR]
    ]
    assert len(base_blocks) == 1, "expected one theme-independent anchor block"
    return base_blocks[0]


def _host_declarations(theme: str) -> dict[str, str]:
    """What the adapter declares on an anchor carrying ``data-theme="<theme>"``."""
    themed_anchor = f'{_HOST_ANCHOR}[data-theme="{theme}"]'
    declarations: dict[str, str] = {}
    for selectors, rule_declarations in _canopy_host_rules():
        if _HOST_ANCHOR in selectors or themed_anchor in selectors:
            declarations.update(rule_declarations)
    return declarations


def _mix(first: _Rgba, second: _Rgba, second_share: float) -> _Rgba:
    """``color-mix(in srgb, first, second <share>)`` of two opaque colors."""
    assert first[3] == 1.0 and second[3] == 1.0, "color-mix inputs must be opaque here"
    red, green, blue = (
        (1.0 - second_share) * a + second_share * b
        for a, b in zip(first[:3], second[:3], strict=True)
    )
    return (red, green, blue, 1.0)


def _composite(top: _Rgba, bottom: _Rgba) -> _Rgba:
    """A translucent ``top`` painted over an opaque ``bottom``, channels unrounded."""
    assert bottom[3] == 1.0, "the surface under a tint must be opaque"
    alpha = top[3]
    red, green, blue = (
        alpha * a + (1.0 - alpha) * b for a, b in zip(top[:3], bottom[:3], strict=True)
    )
    return (red, green, blue, 1.0)


def _evaluate(expression: str, tokens: dict[str, str]) -> _Rgba:
    """Resolve an adapter color value against one theme's token declarations."""
    expression = expression.strip()
    if expression in _NAMED_COLORS:
        return _NAMED_COLORS[expression]
    reference = _VAR_REFERENCE_PATTERN.fullmatch(expression)
    if reference:
        color = parse_color(tokens[reference.group(1)])
        assert color is not None, f"--{reference.group(1)} is not a color"
        return (float(color.red), float(color.green), float(color.blue), color.alpha)
    mix = _COLOR_MIX_PATTERN.fullmatch(expression)
    assert mix, f"cannot evaluate adapter value {expression!r}"
    parts = [_MIX_PART_PATTERN.fullmatch(part.strip()) for part in mix.groups()]
    assert all(parts), f"color-mix needs both percentages spelled out: {expression!r}"
    (first, first_share), (second, second_share) = (
        (part.group(1), float(part.group(2))) for part in parts if part
    )
    assert first_share + second_share == 100.0, f"color-mix shares must sum to 100: {expression!r}"
    return _mix(_evaluate(first, tokens), _evaluate(second, tokens), second_share / 100.0)


def _relative_luminance(color: _Rgba) -> float:
    """``validate.relative_luminance`` for unrounded channels (same formula)."""
    assert color[3] == 1.0, "contrast is only defined between opaque colors"

    def channel(value: float) -> float:
        c = value / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(value) for value in color[:3])
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _float_contrast_ratio(foreground: _Rgba, background: _Rgba) -> float:
    """``validate.contrast_ratio`` for unrounded channels (same formula)."""
    first, second = _relative_luminance(foreground), _relative_luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def _adapter_pairs(theme: str, tokens: dict[str, str]) -> dict[str, tuple[_Rgba, _Rgba]]:
    """Foreground and background of every gated pair, read through the adapter."""
    declarations = _host_declarations(theme)

    def host(name: str) -> _Rgba:
        return _evaluate(declarations[_HOST_PREFIX + name], tokens)

    secondary_mix = _mix(host("bg-elevated"), host("text-primary"), _SECONDARY_MIX_SHARE)
    selection = _composite(host("accent-tint"), host("bg-panel"))
    return {
        "text-primary/bg-elevated": (host("text-primary"), host("bg-elevated")),
        "text-primary/secondary-mix": (host("text-primary"), secondary_mix),
        "text-secondary/bg-secondary": (host("text-secondary"), host("bg-field")),
        "on-accent/accent": (host("on-accent"), host("accent")),
        "on-accent/accent-hover": (host("on-accent"), host("accent-hover")),
        "accent/selection": (host("accent"), selection),
        "on-accent/text-muted": (host("on-accent"), host("text-muted")),
        "on-accent/status-success": (host("on-accent"), host("status-success")),
    }


def test_canopy_host_css_is_hook_clean_on_disk() -> None:
    content = _CANOPY_HOST_CSS.read_text(encoding="utf-8")

    assert "\r" not in content
    assert content.endswith("\n") and not content.endswith("\n\n")
    for line in content.split("\n"):
        assert line == line.rstrip(), f"trailing whitespace on {line!r}"


def test_canopy_host_references_exist_in_every_theme() -> None:
    """Each token the adapter reads is declared by all 8 theme blocks, or only at root.

    The first block of ``tokens.css`` is both ``:root`` and the ``dark``
    theme, so a name declared there and nowhere else is theme-independent
    and resolves under every theme. A name some theme blocks declare and
    others do not would resolve to the ``dark`` value in the themes that
    lack it: a wrong color rather than a missing one.
    """
    css = _CSS_COMMENT_PATTERN.sub("", _CANOPY_HOST_CSS.read_text(encoding="utf-8"))
    referenced = set(re.findall(r"var\(--([a-zA-Z0-9-]+)", css))
    blocks = _token_blocks()

    unresolved = sorted(
        name
        for name in referenced
        if not all(name in blocks[theme] for theme in THEME_IDS)
        and not (
            name in blocks["dark"]
            and not any(name in blocks[theme] for theme in THEME_IDS if theme != "dark")
        )
    )

    assert referenced, "canopy-host.css references no token"
    assert unresolved == [], (
        f"canopy-host.css reads --{{{', '.join(unresolved)}}}, "
        "declared neither in all 8 theme blocks nor at root only"
    )


def test_canopy_host_declares_every_contract_name_and_the_marker() -> None:
    """The theme-independent block publishes exactly the contract, plus version 1."""
    declared = _host_base_block()
    expected = {_HOST_PREFIX + name for name in HOST_CONTRACT_NAMES} | {_HOST_MARKER}

    assert len(HOST_CONTRACT_NAMES) == 35
    assert set(declared) == expected, (
        f"missing={sorted(expected - set(declared))}, "
        f"not in the contract={sorted(set(declared) - expected)}"
    )
    assert declared[_HOST_MARKER] == "1"
    assert "--canopy-host-contract: 1;" in _CANOPY_HOST_CSS.read_text(encoding="utf-8")


#: The host names whose value is a shared token, each with that token: the
#: pane header's height and the three scale tokens the CANOPY bar reads. Each
#: must be exactly ``var(--<token>)``, never a copy of the token's value.
_TOKEN_SOURCED_HOST_NAMES: dict[str, str] = {
    "bar-h": "wt-pane-head-height",
    "weight-medium": "weight-medium",
    "shadow-dropdown": "shadow-dropdown",
    "z-dropdown": "z-dropdown",
}

#: The bar names appended to host contract v1, in the contract's order.
_BAR_HOST_NAMES: tuple[str, ...] = (
    "bar-h",
    "bar-icon",
    "tab-min-w",
    "tab-max-w",
    "tab-underline",
    "weight-medium",
    "shadow-dropdown",
    "z-dropdown",
)


def test_canopy_host_bar_names_close_the_contract_in_order() -> None:
    """The eight bar names are the contract's last eight, in the contract's order."""
    assert HOST_CONTRACT_NAMES[-len(_BAR_HOST_NAMES) :] == _BAR_HOST_NAMES


def test_canopy_host_token_sourced_names_are_plain_references_to_root_tokens() -> None:
    """Each token-sourced host name is ``var(--<token>)`` of a token the root block declares."""
    declared = _host_base_block()
    root = _token_blocks()["dark"]
    offenders = []
    for name, token in _TOKEN_SOURCED_HOST_NAMES.items():
        value = declared.get(_HOST_PREFIX + name)
        if value != f"var(--{token})":
            offenders.append(f"--{_HOST_PREFIX}{name}: {value!r} (expected var(--{token}))")
        elif token not in root:
            offenders.append(f"--{_HOST_PREFIX}{name}: --{token} is not declared at root")
    assert offenders == [], offenders
    # CANOPY's own lengths: OSPREY has no token for them.
    assert declared[_HOST_PREFIX + "tab-min-w"] == "96px"
    assert declared[_HOST_PREFIX + "tab-max-w"] == "220px"
    assert declared[_HOST_PREFIX + "tab-underline"] == "2px"
    assert declared[_HOST_PREFIX + "bar-icon"] == "14px"
    assert declared[_HOST_PREFIX + "control-h-xs"] == "24px"


def test_canopy_host_values_are_token_references_mixes_or_lengths() -> None:
    offenders = []
    for selectors, declarations in _canopy_host_rules():
        for name, value in declarations.items():
            if name == _HOST_MARKER or _LENGTH_PATTERN.fullmatch(value):
                continue
            if _VAR_REFERENCE_PATTERN.fullmatch(value):
                continue
            mix = _COLOR_MIX_PATTERN.fullmatch(value)
            if mix and all(
                (part_match := _MIX_PART_PATTERN.fullmatch(part.strip()))
                and (
                    part_match.group(1) in _NAMED_COLORS
                    or _VAR_REFERENCE_PATTERN.fullmatch(part_match.group(1))
                )
                for part in mix.groups()
            ):
                continue
            offenders.append(f"{', '.join(selectors)} {{ --{name}: {value} }}")

    assert offenders == []


def test_canopy_host_theme_selectors_are_compound_on_the_anchor() -> None:
    """No bare or descendant ``[data-theme]``: it would miss the anchor or leak past it."""
    css = _CSS_COMMENT_PATTERN.sub("", _CANOPY_HOST_CSS.read_text(encoding="utf-8"))
    positions = [match.start() for match in re.finditer(r"\[data-theme=", css)]

    assert positions, "canopy-host.css has no per-theme rule"
    for position in positions:
        assert css[:position].endswith(_HOST_ANCHOR), (
            f"[data-theme= not directly after {_HOST_ANCHOR}: "
            f"{css[max(0, position - 40) : position + 30]!r}"
        )
    for selectors, _ in _canopy_host_rules():
        for selector in selectors:
            assert selector == _HOST_ANCHOR or re.fullmatch(
                re.escape(_HOST_ANCHOR) + _THEME_SELECTOR_PATTERN.pattern, selector
            ), f"unexpected selector {selector!r}"


def test_canopy_host_theme_rules_are_the_five_accent_hover_exceptions() -> None:
    themed: dict[str, set[str]] = {}
    for selectors, declarations in _canopy_host_rules():
        for selector in selectors:
            match = _THEME_SELECTOR_PATTERN.search(selector)
            if match:
                assert match.group(1) not in themed, f"{selector} is declared twice"
                themed[match.group(1)] = set(declarations)

    assert set(themed) == _ACCENT_HOVER_EXCEPTIONS
    assert all(names == {_HOST_PREFIX + "accent-hover"} for names in themed.values()), themed


@pytest.mark.parametrize("theme", THEME_IDS)
def test_canopy_host_accent_hover_differs_from_accent(theme: str) -> None:
    """Hover on a primary button must be visible in every theme."""
    tokens = _token_blocks()[theme]
    declarations = _host_declarations(theme)

    accent = _evaluate(declarations[_HOST_PREFIX + "accent"], tokens)
    accent_hover = _evaluate(declarations[_HOST_PREFIX + "accent-hover"], tokens)

    assert accent_hover != accent


def test_float_contrast_helper_agrees_with_the_generator() -> None:
    """The unrounded-channel helper must be OSPREY's own formula, not a variant of it."""
    for theme, tokens in _token_blocks().items():
        foreground = parse_color(tokens["text-primary"])
        background = parse_color(tokens["bg-elevated"])
        assert foreground is not None and background is not None

        expected = contrast_ratio(foreground, background)
        measured = _float_contrast_ratio(
            _evaluate("var(--text-primary)", tokens), _evaluate("var(--bg-elevated)", tokens)
        )

        assert measured == pytest.approx(expected, abs=1e-12), theme


def test_canopy_host_contrast_pairs_meet_the_family_gates() -> None:
    """Every adapter pair clears its family's minimum, except the pinned shortfalls.

    Each pair takes its minimum from the gate tuple OSPREY applies to the
    theme's own family, by role. The measured failures must equal
    ``KNOWN_SHORTFALLS`` in both directions, each ratio within the
    tolerance.
    """
    tree = load_token_tree(DEFAULT_TOKENS_DIR)
    blocks = _token_blocks()
    assert set(tree.theme_metadata) == set(THEME_IDS)

    failures: dict[tuple[str, str], float] = {}
    measured_pairs = 0
    for theme in THEME_IDS:
        gates = gates_for_family(tree.theme_metadata[theme].get("family"))
        pairs = _adapter_pairs(theme, blocks[theme])
        assert set(pairs) == set(_CONTRAST_PAIR_ROLES)
        for label, (foreground, background) in pairs.items():
            role = _CONTRAST_PAIR_ROLES[label]
            minimum = next(gate.minimum for gate in gates if gate.foreground == role)
            ratio = _float_contrast_ratio(foreground, background)
            measured_pairs += 1
            if ratio < minimum:
                failures[(theme, label)] = ratio

    assert measured_pairs == len(THEME_IDS) * len(_CONTRAST_PAIR_ROLES)
    new = {key: round(failures[key], 2) for key in sorted(set(failures) - set(KNOWN_SHORTFALLS))}
    fixed = sorted(set(KNOWN_SHORTFALLS) - set(failures))
    assert not new and not fixed, (
        f"pairs below their family minimum that are not pinned: {new}; "
        f"pinned shortfalls that now pass (delete the entry): {fixed}"
    )
    for key, pinned in KNOWN_SHORTFALLS.items():
        assert failures[key] == pytest.approx(pinned, abs=_SHORTFALL_TOLERANCE), key
