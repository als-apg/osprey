"""One-header contract: the host tile bar is the only header a panel has.

A panel page has no chrome of its own and no theme control: the hub's tile bar
owns identity and window management, and a panel's controls reach that bar via
the header-contribution contract (design_system/static/js/header-contrib.js).
So no panel stylesheet has anything to hide when framed, and a
``body.embedded`` rule in one means a standalone branch has come back.
"""

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "osprey"
INTERFACES = SRC / "interfaces"

PANEL_PAGES = [
    "ariel/static/css/layout.css",
    "artifacts/static/css/gallery.css",
    "channel_finder/static/css/channel-finder.css",
    "health/static/dashboard.css",
    "okf_panel/static/style.css",
    "bluesky_web/panels/bluesky/panel.css",
    "design_system/panels/reference/index.html",
]


@pytest.mark.parametrize("page", PANEL_PAGES)
def test_panel_page_has_no_framed_only_rule(page: str) -> None:
    assert "body.embedded" not in (INTERFACES / page).read_text(encoding="utf-8")


def test_dispatch_dashboard_has_no_framed_only_rule() -> None:
    html = (SRC / "dispatch" / "dashboard.html").read_text(encoding="utf-8")
    assert "body.embedded {" not in html
    assert "body.embedded ." not in html
    assert "applyEmbedded()" in html, "header-contrib.js reads the class applyEmbedded() sets"
