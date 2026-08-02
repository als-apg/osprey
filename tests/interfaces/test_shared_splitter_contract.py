"""Shared-splitter contract: one divider implementation, every resizable pane.

The OKF knowledge panel, the artifact gallery's browse view, the PLAN panel,
the LATTICE control sidebar, the settings drawer, and the event dashboard's
layout zones all put a draggable divider between two panes. Those were once
four hand-copied implementations, and they drifted — the grip pill was a child
element in one and a ``::after`` in another, the hover treatment disagreed, and
two of them never grew keyboard or ARIA support at all — because nothing failed
when a copy stopped matching. This is that missing failure.

The contract is static and deliberately shallow: each host's markup must carry
the shared ``.osprey-splitter`` class on a proper ARIA separator, each host's
script must drive it from the design system's ``splitter.js``, and no host may
re-declare the shared visual. Behaviour itself is covered once, in
``tests/interfaces/design_system/splitter.test.mjs``; the surfaces are driven
live in ``tests/interfaces/design_system/test_splitter_surfaces.py``.

Two rules here exist because of specific bugs this suite is meant to prevent:

- **Every** splitter tag is checked, not just the first. The gallery ships two
  handles (one per orientation), and a first-match-only assertion would leave
  whichever one happened to be second entirely unguarded.
- ``aria-orientation`` is checked against a per-host expected set rather than
  hardcoded to ``vertical``. ``splitter.js`` sets the attribute at runtime from
  its ``axis`` option, but a static guard can only see markup — so hosts
  declare it too, which also means the separator is correctly labelled before
  the module loads.
"""

import re
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parents[2]
INTERFACES = REPO / "src" / "osprey" / "interfaces"

DESIGN_SYSTEM_CSS = INTERFACES / "design_system" / "static" / "css" / "base.css"


class Host(NamedTuple):
    """One resizable-pane surface and where its three halves live."""

    label: str
    #: Markup carrying the ``.osprey-splitter`` element(s), relative to the repo root.
    html: str
    #: Script calling ``initSplitter``, relative to the repo root.
    js: str
    #: Every ``aria-orientation`` value this host is allowed to declare.
    orientations: frozenset[str]
    #: Stylesheets that must not restate the shared visual, relative to the repo root.
    styles: tuple[str, ...]


_IFACE = "src/osprey/interfaces"

#: Every resizable-pane surface in the product.
SPLITTER_HOSTS = [
    Host(
        label="okf",
        html=f"{_IFACE}/okf_panel/static/index.html",
        js=f"{_IFACE}/okf_panel/static/js/resize.js",
        orientations=frozenset({"vertical"}),
        styles=(f"{_IFACE}/okf_panel/static/style.css",),
    ),
    Host(
        label="artifacts",
        html=f"{_IFACE}/artifacts/static/index.html",
        js=f"{_IFACE}/artifacts/static/js/browse-layout.js",
        # Two handles: the side-by-side split and the stacked-mode split.
        orientations=frozenset({"vertical", "horizontal"}),
        styles=(f"{_IFACE}/artifacts/static/css/gallery.css",),
    ),
    Host(
        label="plan",
        html=f"{_IFACE}/bluesky_panels/panels/plan/index.html",
        js=f"{_IFACE}/bluesky_panels/panels/plan/panel.js",
        orientations=frozenset({"vertical"}),
        styles=(f"{_IFACE}/bluesky_panels/panels/plan/panel.css",),
    ),
    Host(
        label="lattice",
        html=f"{_IFACE}/lattice_dashboard/static/index.html",
        js=f"{_IFACE}/lattice_dashboard/static/js/ui.js",
        orientations=frozenset({"vertical"}),
        styles=(f"{_IFACE}/lattice_dashboard/static/dashboard.css",),
    ),
    Host(
        label="drawer",
        html=f"{_IFACE}/web_terminal/static/index.html",
        js=f"{_IFACE}/design_system/static/js/components/osprey-drawer.js",
        orientations=frozenset({"vertical"}),
        styles=(f"{_IFACE}/web_terminal/static/css/drawer.css",),
    ),
    Host(
        label="events",
        # The event dashboard is a single self-contained file: markup, script
        # and stylesheet are all the same path.
        html="src/osprey/dispatch/dashboard.html",
        js="src/osprey/dispatch/dashboard.html",
        # A sidebar split (vertical) plus two stacked zone splits (horizontal).
        orientations=frozenset({"vertical", "horizontal"}),
        styles=(),
    ),
]

#: Declarations that define the shared grip-pill look. A host restating any of
#: these is re-forking the visual, which is exactly the drift this guards.
FORKED_VISUAL_PROPERTIES = ("cursor: col-resize", "cursor: row-resize", "touch-action: none")

#: Identifiers from the hand-rolled implementations this feature replaced. Any
#: of them reappearing means a surface has grown a private splitter again.
RETIRED_IMPLEMENTATIONS = (
    "wireVerticalDrag",
    "wireResizers",
    "_beginResizeDrag",
    "_teardownResizeDragListeners",
    "vhandle",
    "hhandle",
)


def _markup(path: str) -> str:
    """The file's real markup: whitespace-normalized, HTML comments removed.

    Comments matter here — every one of these dividers is documented in a
    comment that names the shared class, which would otherwise satisfy these
    assertions without any element existing.
    """
    raw = (REPO / path).read_text(encoding="utf-8")
    return " ".join(re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL).split())


@pytest.mark.parametrize("host", SPLITTER_HOSTS, ids=lambda h: h.label)
def test_splitter_markup_uses_the_shared_class(host: Host) -> None:
    assert "osprey-splitter" in _markup(host.html), (
        f"{host.label}: the pane divider must carry the shared .osprey-splitter class "
        "(design_system/static/css/base.css), not a private copy of the grip pill"
    )


@pytest.mark.parametrize("host", SPLITTER_HOSTS, ids=lambda h: h.label)
def test_every_splitter_is_a_keyboard_reachable_aria_separator(host: Host) -> None:
    """All of them — a first-match-only check leaves later handles unguarded."""
    html = _markup(host.html)
    tags = re.findall(r"<[^>]*osprey-splitter[^>]*>", html)
    assert tags, f"{host.label}: no element carries .osprey-splitter"

    seen_orientations = set()
    for tag in tags:
        assert 'role="separator"' in tag, (
            f"{host.label}: splitter must be role=separator, got {tag}"
        )
        assert "tabindex=" in tag, (
            f"{host.label}: splitter must be focusable — Arrow-key resize is the only way "
            f"to move this divider without a pointer, got {tag}"
        )
        assert "aria-label=" in tag, f"{host.label}: splitter needs an accessible name, got {tag}"

        orientation = re.search(r'aria-orientation="([^"]+)"', tag)
        assert orientation is not None, f"{host.label}: splitter needs aria-orientation, got {tag}"
        assert orientation.group(1) in host.orientations, (
            f"{host.label}: aria-orientation={orientation.group(1)!r} is not one of "
            f"{sorted(host.orientations)} — the markup and splitter.js's axis option disagree"
        )
        seen_orientations.add(orientation.group(1))

    assert seen_orientations == set(host.orientations), (
        f"{host.label}: declared orientations {sorted(host.orientations)} but the markup only "
        f"ships {sorted(seen_orientations)}"
    )


@pytest.mark.parametrize("host", SPLITTER_HOSTS, ids=lambda h: h.label)
def test_splitter_behaviour_comes_from_the_design_system(host: Host) -> None:
    js = (REPO / host.js).read_text(encoding="utf-8")

    assert "/design-system/js/splitter.js" in js, (
        f"{host.label}: drag/clamp/persist/keyboard behaviour must come from the shared "
        "module, not a local reimplementation"
    )
    assert "initSplitter" in js, f"{host.label}: must call initSplitter()"


@pytest.mark.parametrize("host", SPLITTER_HOSTS, ids=lambda h: h.label)
def test_hosts_do_not_restate_the_shared_visual(host: Host) -> None:
    """A host's own stylesheet must not re-declare the grip-pill properties."""
    for style in host.styles:
        css = (REPO / style).read_text(encoding="utf-8")
        for prop in FORKED_VISUAL_PROPERTIES:
            assert prop not in css, (
                f"{host.label}: {Path(style).name} restates '{prop}', which belongs to "
                ".osprey-splitter in the design system's base.css"
            )


def test_no_hand_rolled_drag_implementation_survives() -> None:
    """The retired implementations must not reappear anywhere in the product."""
    offenders: list[str] = []
    for path in (REPO / "src" / "osprey").rglob("*"):
        if path.suffix not in {".js", ".html", ".css"} or "vendor" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in RETIRED_IMPLEMENTATIONS:
            if name in text:
                offenders.append(f"{path.relative_to(REPO)}: {name}")

    assert not offenders, (
        "hand-rolled splitter code is back — every divider must go through "
        "design_system/static/js/splitter.js:\n  " + "\n  ".join(offenders)
    )


def test_the_shared_visual_actually_exists() -> None:
    """Guards the other direction: the class every host depends on is real."""
    css = " ".join(DESIGN_SYSTEM_CSS.read_text(encoding="utf-8").split())

    assert ".osprey-splitter {" in css
    for prop in FORKED_VISUAL_PROPERTIES:
        assert prop in css, f"base.css .osprey-splitter is missing '{prop}'"
    # Both geometries, so a y-axis divider is not silently left with a
    # col-resize cursor and an upright pill.
    assert '.osprey-splitter[aria-orientation="horizontal"]' in css
    # The interaction states are the whole point of the shared look.
    assert ".osprey-splitter.dragging" in css
    assert ".osprey-splitter:focus-visible" in css


def test_a_drag_can_never_animate_the_property_it_drives() -> None:
    """The structural fix for the defect that motivated this whole feature.

    A ``transition`` on the property a drag is driving makes the pane trail the
    cursor by the transition duration — the browser eases toward a target the
    pointer has already left. Deleting the offending declarations would fix it
    once; asserting the suppression rule exists makes every future transition
    on a splitter pane harmless by construction, which is the difference
    between a fix and a guarantee.
    """
    css = " ".join(DESIGN_SYSTEM_CSS.read_text(encoding="utf-8").split())

    assert "[data-splitter-dragging] { transition: none !important; }" in css, (
        "base.css must suppress transitions on a pane being dragged; without it a host "
        "that transitions width/height reintroduces the trailing-cursor lag silently"
    )
