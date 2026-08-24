"""Browser tests: where the rail's utility cluster actually lands.

``#panel-utility`` (Documentation + Feedback) is pinned to the far end of the
rail by nothing but ``margin-top: auto`` in the left column and
``margin-left: auto`` under a top rail — ``.panel-rail`` has no ``flex: 1``, so
those two declarations are the whole mechanism. A stylesheet edit that drops
either one leaves the pair riding directly under the panel tabs, where it reads
as two more panel tabs. That is a geometry fact: no unit test can see it,
because only a real layout engine resolves ``auto`` margins.

Coverage:

  (1) left mode (default) — the cluster's bottom sits on the rail region's
      bottom content edge, and the whole cluster is below the last panel tab.
  (2) ``?rail=top`` — the cluster's right edge sits on the region's right
      content edge, and every visible button fits the 40px strip.
  (3) both modes — each control's label fits its cell, rather than being
      quietly ellipsised down to an abbreviation nobody chose.

Run:
    .venv/bin/pytest tests/interfaces/web_terminal/test_feedback_rail_browser.py -v

Skips cleanly when the chromium headless binary is not installed.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from tests.interfaces._panel_launch import publish_artifact_url
from tests.interfaces.conftest import _apply_all, _run_app_server

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

try:
    from playwright.sync_api import Browser, Page, expect

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = [pytest.mark.browser, pytest.mark.slow]

#: Sub-pixel slack for a "these two edges coincide" comparison. Chromium
#: reports fractional device-pixel geometry, so an exact equality on box edges
#: is a flake waiting to happen; a whole pixel is far below the failure this
#: suite exists to catch (a lost ``margin: auto`` moves the cluster by hundreds).
_EDGE_TOLERANCE_PX = 1.0


@contextmanager
def _hub_server(workspace_dir: Path) -> Iterator[str]:
    """Launch a real web-terminal hub with just the artifacts panel enabled.

    The same patch set every hub browser suite uses: the companion backends are
    bypassed and the artifacts panel reports the standard fallback URL, so the
    rail renders one real panel tab without a backend process.

    Yields:
        The hub's base URL.
    """
    patches = [
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=({"artifacts"}, [], None),
        ),
        patch(
            "osprey.interfaces.web_terminal.app._launch_panel_server",
            side_effect=publish_artifact_url(),
        ),
    ]
    with _apply_all(patches):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command=["echo", "hello"])
        with _run_app_server(app) as base_url:
            yield base_url


def _open_hub_page(browser: Browser, url: str) -> Page:
    """Open a hub URL and wait for the rail entries AND the utility cluster.

    The cluster is static markup, but the pin only becomes measurable once
    ``createRail()`` has filled the nav above it — before that the column has a
    different amount of free space to absorb.
    """
    page = browser.new_page()
    page.goto(url, wait_until="domcontentloaded")
    expect(page.locator('button.panel-rail-button[data-panel-id="artifacts"]')).to_be_attached(
        timeout=10_000
    )
    expect(page.locator("#panel-feedback-btn")).to_be_visible(timeout=10_000)
    return page


def _box(page: Page, selector: str) -> dict[str, float]:
    """The element's bounding box, with ``right``/``bottom`` filled in."""
    box = page.locator(selector).bounding_box()
    assert box is not None, f"{selector} has no layout box"
    return {**box, "right": box["x"] + box["width"], "bottom": box["y"] + box["height"]}


def _padding_px(page: Page, selector: str, side: str) -> float:
    """One computed padding edge, in CSS pixels.

    Read rather than hardcoded: the assertions below are about the cluster
    reaching the region's *content* edge, and the region's padding is a
    presentational choice that may legitimately change.
    """
    return float(
        page.evaluate(
            "([sel, side]) => parseFloat("
            "  getComputedStyle(document.querySelector(sel)).getPropertyValue('padding-' + side)"
            ")",
            [selector, side],
        )
    )


def _content_height_px(page: Page, selector: str) -> float:
    """The element's content-box height, in CSS pixels.

    Read for the same reason ``_padding_px`` is: the strip's height is a
    presentational choice, and the claim under test is that a button fits
    *inside whatever the strip is* — not that either one equals 40px.
    """
    return float(
        page.evaluate(
            "(sel) => {"
            "  const el = document.querySelector(sel);"
            "  const cs = getComputedStyle(el);"
            "  return el.clientHeight"
            "    - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);"
            "}",
            selector,
        )
    )


def test_utility_cluster_is_pinned_to_the_bottom_of_the_left_rail(tmp_path, chromium_browser):
    """Left mode: the cluster's bottom edge is the rail region's bottom edge.

    ``margin-top: auto`` on ``.panel-utility`` eats the column's free space.
    Without it the cluster rides under the "+" control instead, so the second
    assertion — a clear gap below the last panel tab — is what distinguishes
    "pinned to the far end" from "next in the list".
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, base_url)
        try:
            # State the mode this test measures. The rail can move out from
            # under a page with no stored pin: app.js couples rail position to
            # the theme family, and a family whose default is "top" would turn
            # every assertion below into an opaque edge mismatch rather than
            # "the page is not in left mode".
            expect(page.locator("html")).to_have_attribute("data-rail-position", "left")

            region = _box(page, ".panel-rail-region")
            cluster = _box(page, "#panel-utility")
            pad_bottom = _padding_px(page, ".panel-rail-region", "bottom")

            content_bottom = region["bottom"] - pad_bottom
            assert abs(cluster["bottom"] - content_bottom) <= _EDGE_TOLERANCE_PX, (
                f"utility cluster bottom {cluster['bottom']} is not on the rail region's "
                f"content bottom {content_bottom} — the margin-top:auto pin is gone"
            )

            assert page.locator("button.panel-rail-button").count() >= 1
            last_tab = _box(page, "button.panel-rail-button >> nth=-1")
            assert cluster["y"] >= last_tab["bottom"], (
                "utility cluster overlaps the panel tabs above it"
            )
            # ...and below the "+" control, which is the last cell of the tab
            # block. Both orderings hold whether or not the pin survives, so
            # they are the ordering half of the contract; the edge assertion
            # above is what proves the cluster reached the far end.
            assert cluster["y"] >= _box(page, "#panel-add")["bottom"], (
                "utility cluster is interleaved with the add-panel control"
            )
        finally:
            page.close()


def test_utility_cluster_is_pinned_to_the_right_of_the_top_rail(tmp_path, chromium_browser):
    """Top mode: the cluster turns the corner — far end becomes far right.

    ``margin-left: auto`` replaces ``margin-top: auto``, and the 52px column
    cell has to shrink to the 40px strip: the strip does not clip, so an
    un-overridden button would hang over the workspace below it.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        page = _open_hub_page(chromium_browser, f"{base_url}/?rail=top")
        try:
            expect(page.locator("html")).to_have_attribute("data-rail-position", "top")

            region = _box(page, ".panel-rail-region")
            cluster = _box(page, "#panel-utility")
            pad_right = _padding_px(page, ".panel-rail-region", "right")

            content_right = region["right"] - pad_right
            assert abs(cluster["right"] - content_right) <= _EDGE_TOLERANCE_PX, (
                f"utility cluster right edge {cluster['right']} is not on the rail region's "
                f"content right edge {content_right} — the margin-left:auto pin is gone"
            )

            strip_height = _content_height_px(page, ".panel-rail-region")

            buttons = page.locator("#panel-utility .panel-utility-btn")
            measured = 0
            for index in range(buttons.count()):
                button = buttons.nth(index)
                box = button.bounding_box()
                if box is None:
                    continue  # the docs anchor is hidden without a configured web.docs_url
                measured += 1
                assert box["height"] <= strip_height + _EDGE_TOLERANCE_PX, (
                    f"utility button {index} is {box['height']}px tall — it hangs below the "
                    f"{strip_height}px strip, which does not clip"
                )
            assert measured >= 1, "no visible utility button to measure"
        finally:
            page.close()


def test_utility_labels_fit_their_cells_uncut(tmp_path, chromium_browser):
    """Both orientations: the label is fully readable, not ellipsised away.

    "FEEDBACK" is the long one, and in the left column it has 62px minus the
    cell's own padding to live in. ``text-overflow: ellipsis`` means a label
    that outgrows its cell degrades silently to "FEEDBA…" rather than
    overflowing visibly — so the failure this catches is one that no layout
    exception and no unit test reports. Only a layout engine knows whether the
    text fits, which is why the check lives here and not in the vitest suite.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()

    with _hub_server(workspace) as base_url:
        for url in (base_url, f"{base_url}/?rail=top"):
            page = _open_hub_page(chromium_browser, url)
            try:
                labels = page.locator("#panel-utility .panel-utility-label")
                assert labels.count() == 2, "the cluster lost a label"

                overflow = page.evaluate(
                    "() => Array.from("
                    "  document.querySelectorAll('#panel-utility .panel-utility-label')"
                    ").filter((el) => el.scrollWidth > el.clientWidth + 1)"
                    " .map((el) => el.textContent)"
                )
                assert overflow == [], f"utility labels are cut off at {url}: {overflow}"
            finally:
                page.close()
