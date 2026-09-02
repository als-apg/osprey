"""Shared info-tip contract: one "?" disc and bubble, every host.

The BLUESKY panel's queue controls and the control-target popover's machine
names each explain themselves behind a small round "?"/"i" that opens a bubble
on hover and on keyboard focus. Before the design system carried this, each
host drew its own — and a third and fourth variant (a click-toggled paragraph
under the strip, an inline block under a form row) had already appeared.
Nothing failed when a copy stopped matching. This is that missing failure.

The contract is static and deliberately shallow: ``base.css`` must carry the
shared classes with both reveal states, each host's markup or script must put
the shared classes on its disc and bubble, and no host stylesheet may restate
the shared geometry. Two rules encode the two ways this idiom has drifted:

- **The bubble must be reachable by keyboard.** ``:focus-visible`` is checked
  alongside ``:hover``; a hover-only tip is how the native ``title`` bubble's
  mouse-only failing comes back under a new name.
- **No host may keep a private ``display`` toggle for its bubble.** The
  click-toggle variant lived on ``aria-expanded`` and a ``hidden`` paragraph;
  a host that re-grows one has left the shared idiom, whatever its class name.
"""

import re
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parents[2]
INTERFACES = REPO / "src" / "osprey" / "interfaces"

DESIGN_SYSTEM_CSS = INTERFACES / "design_system" / "static" / "css" / "base.css"

DISC = "osprey-info"
TIP = "osprey-tip"


class Host(NamedTuple):
    """One surface that explains a control behind the shared disc."""

    label: str
    #: Markup or script that builds the disc and bubble, relative to the repo root.
    source: str
    #: Stylesheets that must not restate the shared visual, relative to the repo root.
    styles: tuple[str, ...]


_IFACE = "src/osprey/interfaces"

#: Every surface drawing the shared info tip.
INFO_TIP_HOSTS = [
    Host(
        label="bluesky-queue",
        source=f"{_IFACE}/bluesky_web/panels/bluesky/index.html",
        styles=(f"{_IFACE}/bluesky_web/panels/bluesky/panel.css",),
    ),
    Host(
        label="control-target",
        source=f"{_IFACE}/web_terminal/static/js/control-target-popover.js",
        styles=(f"{_IFACE}/web_terminal/static/css/terminal.css",),
    ),
]


@pytest.fixture(scope="module")
def base_css() -> str:
    return DESIGN_SYSTEM_CSS.read_text()


def _rule_bodies(css: str, selector_pattern: str) -> list[str]:
    """Every declaration block whose selector list matches ``selector_pattern``.

    Comments are stripped first: a host stylesheet is allowed to *mention* the
    shared class in prose, and the selector text before a ``{`` would
    otherwise carry that prose along.
    """
    bare = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return [
        m.group(2)
        for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", bare)
        if re.search(selector_pattern, m.group(1), flags=re.M)
    ]


# --- The design system carries the component ---------------------------------


def test_base_css_defines_the_disc_and_the_bubble(base_css: str) -> None:
    assert _rule_bodies(base_css, rf"^\s*\.{DISC}\s*$"), "base.css lost .osprey-info"
    tip = _rule_bodies(base_css, rf"^\s*\.{TIP}\s*$")
    assert tip, "base.css lost .osprey-tip"
    # The bubble is a real positioned element that stays out of the flow.
    assert "position: absolute" in tip[0]
    assert "display: none" in tip[0]


def test_bubble_is_revealed_on_hover_and_on_keyboard_focus(base_css: str) -> None:
    reveal = re.search(
        rf"\.{DISC}:hover \.{TIP},\s*\.{DISC}:focus-visible \.{TIP}\s*\{{([^}}]*)\}}",
        base_css,
    )
    assert reveal, "the shared reveal rule must list :hover AND :focus-visible"
    assert "display: flex" in reveal.group(1)


def test_bubble_shares_the_fleet_tokens_only(base_css: str) -> None:
    """The bubble's surface, border and type come from tokens, never literals."""
    tip = _rule_bodies(base_css, rf"^\s*\.{TIP}\s*$")[0]
    for prop in ("background", "color", "font-family", "font-size", "border-radius", "box-shadow"):
        value = re.search(rf"\b{prop}:\s*([^;]+);", tip)
        assert value, f".osprey-tip must set {prop}"
        assert "var(--" in value.group(1), (
            f".osprey-tip {prop} must be a token, got {value.group(1)!r}"
        )


# --- Every host uses it, and none restates it ---------------------------------


@pytest.mark.parametrize("host", INFO_TIP_HOSTS, ids=[h.label for h in INFO_TIP_HOSTS])
def test_host_puts_the_shared_classes_on_its_disc_and_bubble(host: Host) -> None:
    source = (REPO / host.source).read_text()
    assert DISC in source, f"{host.label}: the disc must carry .{DISC}"
    assert TIP in source, f"{host.label}: the bubble must carry .{TIP}"
    assert 'role="tooltip"' in source or "'role', 'tooltip'" in source, (
        f"{host.label}: the bubble must be a role=tooltip element"
    )


@pytest.mark.parametrize("host", INFO_TIP_HOSTS, ids=[h.label for h in INFO_TIP_HOSTS])
def test_host_keeps_no_private_open_state(host: Host) -> None:
    """The tip is hover/focus only — no host may click-toggle its bubble."""
    source = (REPO / host.source).read_text()
    disc_tags = [m.group(0) for m in re.finditer(r"<button[^>]*\b" + DISC + r"\b[^>]*>", source)]
    for tag in disc_tags:
        assert "aria-expanded" not in tag, (
            f"{host.label}: the disc must not carry aria-expanded: {tag}"
        )
        assert "aria-controls" not in tag, f"{host.label}: the disc controls nothing: {tag}"


@pytest.mark.parametrize("host", INFO_TIP_HOSTS, ids=[h.label for h in INFO_TIP_HOSTS])
def test_host_stylesheets_do_not_restate_the_shared_visual(host: Host) -> None:
    for rel in host.styles:
        css = (REPO / rel).read_text()
        for cls in (DISC, TIP):
            restated = _rule_bodies(css, rf"(^|[\s,])\.{cls}(\s|,|$|::)")
            assert not restated, f"{rel} restates .{cls}; the shared visual lives in base.css"
