"""Shared-splitter contract: one divider implementation, three two-pane browsers.

The OKF knowledge panel, the artifact gallery's browse view, and the PLAN
panel all put a draggable divider between a browser pane and a detail pane.
Those were once three hand-copied implementations, and they drifted — the
grip pill was a child element in one and a ``::after`` in another, and the
hover treatment disagreed — because nothing failed when a copy stopped
matching. This is that missing failure.

The contract is static and deliberately shallow: each panel's markup must
carry the shared ``.osprey-splitter`` class on a proper ARIA separator, each
panel's script must drive it from the design system's ``splitter.js``, and no
panel may re-declare the shared visual. Behaviour itself is covered once, in
``tests/interfaces/design_system/splitter.test.mjs``.
"""

import re
from pathlib import Path

import pytest

INTERFACES = Path(__file__).resolve().parents[2] / "src" / "osprey" / "interfaces"

DESIGN_SYSTEM_CSS = INTERFACES / "design_system" / "static" / "css" / "base.css"

#: (label, markup file, script file) for every two-pane browser in the product.
SPLITTER_HOSTS = [
    (
        "okf",
        "okf_panel/static/index.html",
        "okf_panel/static/js/resize.js",
    ),
    (
        "artifacts",
        "artifacts/static/index.html",
        "artifacts/static/js/browse-layout.js",
    ),
    (
        "plan",
        "bluesky_panels/panels/plan/index.html",
        "bluesky_panels/panels/plan/panel.js",
    ),
]

#: Declarations that define the shared grip-pill look. A panel restating any of
#: these is re-forking the visual, which is exactly the drift this guards.
FORKED_VISUAL_PROPERTIES = ("cursor: col-resize", "touch-action: none")


def _markup(html_file: str) -> str:
    """The file's real markup: whitespace-normalized, HTML comments removed.

    Comments matter here — every one of these dividers is documented in a
    comment that names the shared class, which would otherwise satisfy these
    assertions without any element existing.
    """
    raw = (INTERFACES / html_file).read_text(encoding="utf-8")
    return " ".join(re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL).split())


@pytest.mark.parametrize(("label", "html_file", "_js_file"), SPLITTER_HOSTS)
def test_splitter_markup_uses_the_shared_class(label: str, html_file: str, _js_file: str) -> None:
    html = _markup(html_file)

    assert "osprey-splitter" in html, (
        f"{label}: the pane divider must carry the shared .osprey-splitter class "
        "(design_system/static/css/base.css), not a private copy of the grip pill"
    )


@pytest.mark.parametrize(("label", "html_file", "_js_file"), SPLITTER_HOSTS)
def test_splitter_is_a_keyboard_reachable_aria_separator(
    label: str, html_file: str, _js_file: str
) -> None:
    html = _markup(html_file)

    # The element carrying the shared class, up to the end of its open tag.
    match = re.search(r"<[^>]*osprey-splitter[^>]*>", html)
    assert match is not None, f"{label}: no element carries .osprey-splitter"
    tag = match.group(0)

    assert 'role="separator"' in tag, f"{label}: splitter must be role=separator, got {tag}"
    assert 'aria-orientation="vertical"' in tag, f"{label}: splitter needs aria-orientation"
    assert "tabindex=" in tag, (
        f"{label}: splitter must be focusable — Arrow-key resize is the only way "
        "to move this divider without a pointer"
    )
    assert "aria-label=" in tag, f"{label}: splitter needs an accessible name"


@pytest.mark.parametrize(("label", "_html_file", "js_file"), SPLITTER_HOSTS)
def test_splitter_behaviour_comes_from_the_design_system(
    label: str, _html_file: str, js_file: str
) -> None:
    js = (INTERFACES / js_file).read_text(encoding="utf-8")

    assert "/design-system/js/splitter.js" in js, (
        f"{label}: drag/clamp/persist/keyboard behaviour must come from the shared "
        "module, not a local reimplementation"
    )
    assert "initSplitter" in js, f"{label}: must call initSplitter()"


@pytest.mark.parametrize(("label", "_html_file", "js_file"), SPLITTER_HOSTS)
def test_panels_do_not_restate_the_shared_visual(label: str, _html_file: str, js_file: str) -> None:
    """A panel's own stylesheet must not re-declare the grip-pill properties."""
    panel_css_dir = (INTERFACES / js_file).parent
    # Panel stylesheets live beside (or one level up from) the panel script.
    candidates = list(panel_css_dir.glob("*.css")) + list(panel_css_dir.parent.rglob("*.css"))

    for css_path in {c for c in candidates if "design_system" not in str(c)}:
        css = css_path.read_text(encoding="utf-8")
        for prop in FORKED_VISUAL_PROPERTIES:
            assert prop not in css, (
                f"{label}: {css_path.name} restates '{prop}', which belongs to "
                ".osprey-splitter in the design system's base.css"
            )


def test_the_shared_visual_actually_exists() -> None:
    """Guards the other direction: the class every panel now depends on is real."""
    css = " ".join(DESIGN_SYSTEM_CSS.read_text(encoding="utf-8").split())

    assert ".osprey-splitter {" in css
    for prop in FORKED_VISUAL_PROPERTIES:
        assert prop in css, f"base.css .osprey-splitter is missing '{prop}'"
    # The interaction states are the whole point of the shared look.
    assert ".osprey-splitter.dragging" in css
    assert ".osprey-splitter:focus-visible" in css
