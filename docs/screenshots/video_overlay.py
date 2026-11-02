"""On-page overlay for the docs videos: cursor dot, caption bar and spotlight.

``OVERLAY_JS`` is registered with ``context.add_init_script`` so it runs in every
document before the page's own scripts. It only acts in the top frame, so panel
iframes never draw a second cursor. The ``window.__demo*`` functions exist
from the first instant; ``document.body`` does not, so calls made before the
elements mount are queued and replayed once they do.

The Python helpers drive those functions from a sync Playwright ``page``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.sync_api import Page

# Playwright evaluates an init script as a plain script, where a bare top-level
# ``return`` is a syntax error; the IIFE wrapper is what lets the guard return.
OVERLAY_JS = """\
(() => {
if (window.top !== window) return;
const queue = [];
let els = null;

function apply(call) {
  const [kind, a, b] = call;
  if (kind === "cursor") {
    els.cursor.style.transform = "translate(" + (a - 9) + "px," + (b - 9) + "px)";
    els.cursor.style.display = "block";
  } else if (kind === "caption") {
    if (a) {
      els.caption.textContent = String(a);
      els.caption.style.display = "block";
    } else {
      els.caption.textContent = "";
      els.caption.style.display = "none";
    }
  } else if (kind === "spotlight") {
    if (a) {
      const pad = 6;
      Object.assign(els.spotlight.style, {
        left: (a.x - pad) + "px", top: (a.y - pad) + "px",
        width: (a.width + 2 * pad) + "px", height: (a.height + 2 * pad) + "px",
        display: "block",
      });
    } else {
      els.spotlight.style.display = "none";
    }
  }
}

function dispatch(call) {
  if (els) apply(call); else queue.push(call);
}

window.__demoCursor = (x, y) => dispatch(["cursor", x, y]);
window.__demoCaption = (text) => dispatch(["caption", text]);
window.__demoSpotlight = (box) => dispatch(["spotlight", box]);

function el(id, css) {
  const node = document.createElement("div");
  node.id = id;
  node.setAttribute("aria-hidden", "true");
  node.style.cssText = css + ";position:fixed;z-index:2147483647;"
    + "pointer-events:none;display:none;";
  return node;
}

function mount() {
  if (els || !document.body) return;
  // Fixed light-on-dark and dark-on-light pairings with their own backgrounds,
  // so each element reads the same in the light and the dark theme.
  const cursor = el("__demo-cursor",
    "left:0;top:0;width:18px;height:18px;border-radius:50%;"
    + "background:rgba(255,196,0,0.85);border:2px solid #111;"
    + "box-shadow:0 0 0 2px rgba(255,255,255,0.9);box-sizing:border-box;"
    + "transition:transform 16ms linear");
  const caption = el("__demo-caption",
    "left:50%;bottom:28px;transform:translateX(-50%);max-width:80%;"
    + "padding:10px 22px;border-radius:10px;background:rgba(17,17,17,0.88);"
    + "color:#fff;font:600 22px/1.35 system-ui,-apple-system,sans-serif;"
    + "text-align:center;box-shadow:0 4px 18px rgba(0,0,0,0.35)");
  // An outline round one element, the rest of the page dimmed by its shadow.
  const spotlight = el("__demo-spotlight",
    "border:3px solid rgba(255,196,0,0.95);border-radius:8px;box-sizing:border-box;"
    + "box-shadow:0 0 0 9999px rgba(0,0,0,0.45)");
  document.body.append(spotlight, cursor, caption);
  els = { cursor, caption, spotlight };
  while (queue.length) apply(queue.shift());
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", mount, { once: true });
} else {
  mount();
}
})();
"""


def set_spotlight(page: Page | Any, box: dict[str, float] | None) -> None:
    """Outline *box*, a viewport rectangle as Playwright's ``bounding_box`` gives it.

    ``None`` clears the outline.
    """
    page.evaluate("(b) => window.__demoSpotlight(b)", box)


# Position of the overlay cursor per page, so each move starts where the last ended.
_cursor_positions: dict[int, tuple[float, float]] = {}


def set_caption(page: Page | Any, text: str | None) -> None:
    """Show ``text`` in the bottom caption bar; ``None`` or ``""`` hides it."""
    page.evaluate("(t) => window.__demoCaption(t)", text)


def move_cursor(page: Page | Any, x: float, y: float, steps: int = 20) -> None:
    """Glide the real mouse and the overlay dot together to ``(x, y)``.

    The move is split into ``steps`` linear increments (at least one); each moves the
    Playwright mouse and redraws the dot, so the recording shows the path and hover
    effects fire along it.
    """
    steps = max(1, int(steps))
    x0, y0 = _cursor_positions.get(id(page), (0.0, 0.0))
    for i in range(1, steps + 1):
        frac = i / steps
        xi = x0 + (x - x0) * frac
        yi = y0 + (y - y0) * frac
        page.mouse.move(xi, yi)
        page.evaluate("([x, y]) => window.__demoCursor(x, y)", [xi, yi])
    _cursor_positions[id(page)] = (float(x), float(y))
