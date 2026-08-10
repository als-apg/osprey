"""Embedded-identity contract: host tile bar = the ONE header a tile has.

When a panel runs inside the web-terminal hub (body.embedded — set by
frame-params.js applyEmbedded()), it must not render its own top bar at all:
the hub's tile bar owns identity and window management, and a panel's real
toolbar controls reach that bar via the header-contribution contract
(design_system/static/js/header-contrib.js). It must also not render a theme
switcher (the hub owns theming). This is a static contract check on the
shipped CSS: each panel's stylesheet must carry the body.embedded rule that
hides the listed element. Standalone pages are unaffected (all rules are
body.embedded-scoped by construction).
"""

from pathlib import Path

import pytest

INTERFACES = Path(__file__).resolve().parents[2] / "src" / "osprey" / "interfaces"

# (css file, selector fragment that must appear in a body.embedded rule)
CONTRACT = [
    # Whole embedded top bars
    ("artifacts/static/css/gallery.css", "body.embedded .header"),
    ("ariel/static/css/layout.css", "body.embedded .header"),
    ("channel_finder/static/css/channel-finder.css", "body.embedded .app-header"),
    ("lattice_dashboard/static/dashboard.css", "body.embedded #topbar"),
    ("health/static/dashboard.css", "body.embedded .hdr"),
    # Theme switchers (hub owns theming)
    ("ariel/static/css/layout.css", "body.embedded osprey-theme-switcher"),
    ("channel_finder/static/css/channel-finder.css", "body.embedded osprey-theme-switcher"),
    ("lattice_dashboard/static/dashboard.css", "body.embedded osprey-theme-switcher"),
    ("okf_panel/static/style.css", "body.embedded osprey-theme-switcher"),
]


@pytest.mark.parametrize(("css_file", "selector"), CONTRACT)
def test_embedded_rule_present(css_file: str, selector: str) -> None:
    css = (INTERFACES / css_file).read_text(encoding="utf-8")
    normalized = " ".join(css.split())
    assert selector in normalized, (
        f"{css_file} must hide '{selector.removeprefix('body.embedded ').strip()}' "
        "when embedded in the web-terminal hub (host tile bar owns identity/theming)"
    )
