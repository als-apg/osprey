"""The screen a docs video starts on, declared per video and applied before recording.

Every :class:`~docs.screenshots.videos.DocVideo` names its :class:`Layout`: the
view (``expert``, the Claude Code terminal; ``simple``, the operator chat), the
left-sidebar entry that is active, and the tiles open beside the console. The
layout is part of what a video is agreed to show, so the take's setup applies
it (:func:`apply`) and fails the take when the page will not take it; no beat
ever implies it.

Sidebar entries and tiles are named by their panel id, the ``data-panel-id``
of the rail row (``artifacts``, ``okf`` for Knowledge, ...). The console's own
tile is implied by the view and never listed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from docs.screenshots.video_probes import ProbeFailed

VIEWS = ("expert", "simple")

APPLY_BUDGET_MS = 30_000


@dataclass(frozen=True)
class Layout:
    """The view, the active sidebar entry and the open tiles a take starts on."""

    view: Literal["expert", "simple"]
    active: str
    """Panel id of the sidebar entry that is active (``data-panel-id``)."""

    tiles: tuple[str, ...]
    """Panel ids of the tiles open beside the console, in any order."""


# Make the layout so: activate the sidebar row unless it already is, close
# every panel tile not listed (through its tab's close button, so the panel
# manager learns of it), then wait until the page shows exactly the layout.
# Resolves to {ok, detail}.
APPLY_JS = """
async ({ active, tiles, budgetMs }) => {
  const frame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));
  const deadline = Date.now() + budgetMs;
  const row = () => document.querySelector(
    `button.panel-rail-button[data-panel-id="${active}"]`);
  const openTiles = () => [...document.querySelectorAll('.tile-tab[data-panel-id^="iframe:"]')]
    .map((t) => t.dataset.panelId.slice('iframe:'.length));
  const state = () => ({
    active: [...document.querySelectorAll('button.panel-rail-button.active')]
      .map((b) => b.dataset.panelId),
    tiles: [...new Set(openTiles())].sort(),
  });
  const wanted = [...new Set(tiles)].sort();
  let clicked = false;
  while (Date.now() < deadline) {
    const r = row();
    if (r && !r.classList.contains('disabled') && !r.classList.contains('active') && !clicked) {
      r.click();
      clicked = true;
    }
    for (const tab of document.querySelectorAll('.tile-tab[data-panel-id^="iframe:"]')) {
      const id = tab.dataset.panelId.slice('iframe:'.length);
      const close = tab.querySelector('.tile-tab-close');
      if (!tiles.includes(id) && close) close.click();
    }
    const now = state();
    if (now.active.length === 1 && now.active[0] === active
        && JSON.stringify(now.tiles) === JSON.stringify(wanted)) {
      return { ok: true, detail: now };
    }
    await frame();
  }
  return { ok: false, detail: state() };
}
"""


def apply(page: Any, layout: Layout, budget_ms: int = APPLY_BUDGET_MS) -> None:
    """Bring the page to *layout*'s sidebar entry and tiles; ProbeFailed ``layout`` if it will not."""
    result = page.evaluate(
        APPLY_JS, {"active": layout.active, "tiles": list(layout.tiles), "budgetMs": budget_ms}
    )
    if not result or not result.get("ok"):
        detail = (result or {}).get("detail")
        raise ProbeFailed(
            "layout",
            f"the page did not take the layout (active {layout.active!r}, tiles "
            f"{list(layout.tiles)}) within {budget_ms} ms; it shows {detail}",
        )


def validate(layout: Any) -> str | None:
    """Why *layout* is not a usable :class:`Layout`, or None when it is."""
    if not isinstance(layout, Layout):
        return "has no layout"
    if layout.view not in VIEWS:
        return f"layout view {layout.view!r} is not one of {', '.join(VIEWS)}"
    if not layout.active:
        return "layout names no active sidebar entry"
    return None
