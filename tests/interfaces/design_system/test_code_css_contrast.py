"""Contrast gate for the read-only source-code colour scheme (``code.css``).

``code.css`` colours each syntax role by mixing a design token toward
``--text-primary``, at a percentage chosen so the role clears WCAG AA in the
*worst* theme. Those percentages are only correct for the palettes they were
derived from, and nothing else in the generator checks them: the WCAG gates in
``validate.py`` cover the semantic tokens themselves, not a hand-written
stylesheet that combines them.

This module is that missing gate. It re-derives every ratio from the committed
``tokens.css`` and ``code.css``, so retuning any theme's accent — or editing a
mix percentage — fails here rather than shipping unreadable code.

It also pins the CSS/JS contract: the token classes the stylesheet colours must
be exactly the kinds ``python-highlight.js`` emits, so neither side can grow a
role the other does not know about.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.interfaces.design_system.generator.validate import (
    RGBColor,
    contrast_ratio,
    parse_color,
)

_STATIC = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "design_system"
    / "static"
)
_TOKENS_CSS = _STATIC / "css" / "tokens.css"
_CODE_CSS = _STATIC / "css" / "code.css"
_HIGHLIGHT_JS = _STATIC / "js" / "python-highlight.js"

#: WCAG AA for body text. Syntax colouring replaces the text colour outright,
#: so each role has to clear the normal-text floor on its own.
_MIN_RATIO = 4.5

#: The surface the highlighted block is painted on (see ``.source`` in the
#: Bluesky panel — the only consumer today, and the darkest/lightest panel
#: surface either way).
_BACKGROUND = "--bg-panel"


def _theme_blocks() -> dict[str, dict[str, str]]:
    """Map each theme name to its custom-property declarations."""
    css = _TOKENS_CSS.read_text()
    blocks: dict[str, dict[str, str]] = {}
    pattern = r'(?::root, \[data-theme="([^"]+)"\]|\[data-theme="([^"]+)"\])\s*\{(.*?)\n\}'
    for match in re.finditer(pattern, css, re.S):
        name = match.group(1) or match.group(2)
        blocks[name] = dict(re.findall(r"(--[\w-]+):\s*([^;]+);", match.group(3)))
    return blocks


def _role_colours() -> dict[str, tuple[str, int, str]]:
    """Map each ``.tok-*`` role to ``(token, mix_percent, mix_partner)``.

    A role declared without ``color-mix()`` is reported as 100% of its token
    mixed with itself, so both forms flow through one code path.
    """
    css = _CODE_CSS.read_text()
    roles: dict[str, tuple[str, int, str]] = {}
    for match in re.finditer(r"\.code-python \.tok-([\w-]+)\s*\{(.*?)\}", css, re.S):
        role, body = match.group(1), match.group(2)
        mixed = re.search(
            r"color:\s*color-mix\(in srgb,\s*var\((--[\w-]+)\)\s*(\d+)%,\s*var\((--[\w-]+)\)\)",
            body,
        )
        if mixed:
            roles[role] = (mixed.group(1), int(mixed.group(2)), mixed.group(3))
            continue
        plain = re.search(r"color:\s*var\((--[\w-]+)\)\s*;", body)
        assert plain, f".tok-{role} declares no resolvable color"
        roles[role] = (plain.group(1), 100, plain.group(1))
    return roles


def _mix(base: RGBColor, partner: RGBColor, percent: int) -> RGBColor:
    """Blend two opaque colours the way CSS ``color-mix(in srgb, …)`` does."""
    weight = percent / 100

    def channel(a: int, b: int) -> int:
        return round(weight * a + (1 - weight) * b)

    return RGBColor(
        red=channel(base.red, partner.red),
        green=channel(base.green, partner.green),
        blue=channel(base.blue, partner.blue),
    )


def _resolve(tokens: dict[str, str], name: str) -> RGBColor:
    raw = tokens.get(name)
    assert raw, f"{name} missing from theme"
    colour = parse_color(raw.strip())
    assert colour, f"{name} is not an opaque colour: {raw!r}"
    return colour


@pytest.mark.unit
@pytest.mark.parametrize("theme", sorted(_theme_blocks()))
def test_every_syntax_role_clears_wcag_aa(theme: str) -> None:
    """Every role in every theme reaches 4.5:1 against the block background."""
    tokens = _theme_blocks()[theme]
    background = _resolve(tokens, _BACKGROUND)

    failures: list[str] = []
    for role, (token, percent, partner) in sorted(_role_colours().items()):
        colour = _mix(_resolve(tokens, token), _resolve(tokens, partner), percent)
        ratio = contrast_ratio(colour, background)
        if ratio < _MIN_RATIO:
            failures.append(f"  .tok-{role}: {token} @ {percent}% -> {ratio:.2f}:1")

    assert not failures, (
        f"code.css roles fall below WCAG AA ({_MIN_RATIO}:1) on theme {theme!r}:\n"
        + "\n".join(failures)
        + "\n\nLower the mix percentage for the role(s) above until the worst "
        "theme clears the floor, and update the table in code.css's header."
    )


@pytest.mark.unit
def test_mix_percentages_are_the_most_hue_each_role_can_keep() -> None:
    """The percentages are maximal — no role is washed out more than needed.

    A too-low percentage is not a contrast bug, so nothing else would catch it;
    it just quietly drains the colour out of the scheme. This pins each role
    within 5 points of the highest percentage its worst theme can carry.
    """
    themes = _theme_blocks()
    slack: list[str] = []
    for role, (token, percent, partner) in sorted(_role_colours().items()):
        if percent == 100:
            continue  # unmixed roles have no percentage to tune
        best = 0
        for candidate in range(100, 0, -1):
            worst = min(
                contrast_ratio(
                    _mix(_resolve(tokens, token), _resolve(tokens, partner), candidate),
                    _resolve(tokens, _BACKGROUND),
                )
                for tokens in themes.values()
            )
            if worst >= _MIN_RATIO:
                best = candidate
                break
        if best - percent > 5:
            slack.append(f"  .tok-{role}: uses {percent}%, could keep {best}%")

    assert not slack, "code.css mixes away more hue than the themes require:\n" + "\n".join(slack)


@pytest.mark.unit
def test_stylesheet_and_highlighter_agree_on_token_kinds() -> None:
    """The roles ``code.css`` colours are exactly the kinds the lexer emits."""
    js = _HIGHLIGHT_JS.read_text()
    typedef = re.search(r"@property \{([^}]*)\} kind", js)
    assert typedef, "python-highlight.js no longer declares its kind union"
    kinds = set(re.findall(r"'([\w-]+)'", typedef.group(1)))
    kinds.discard("plain")  # 'plain' is a bare text node — deliberately unstyled

    assert set(_role_colours()) == kinds, (
        "code.css and python-highlight.js disagree on token kinds. "
        "A kind with no rule renders unstyled; a rule with no kind is dead CSS."
    )
