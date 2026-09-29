"""Page probes for the landing-page demo video recorder.

Small, synchronous checks over Playwright page and frame objects: is the
terminal focused, what does it show, is the Claude Code REPL ready, where is
the plot, which way does its camera look, and is the logbook draft rendered.

Every wait is a ``page.wait_for_timeout`` poll: sync Playwright only pumps its
event loop inside Playwright calls, so a plain sleep would freeze the page.
A probe that gives up raises :class:`ProbeFailed` naming the storyboard step,
which is what the retry loop reports.
"""

from __future__ import annotations

import math
from typing import Any

# Claude Code's first-run folder-trust dialog and its idle REPL, as rendered by
# Claude Code 2.1.282: the "? for shortcuts" footer, or — where a deployment's
# status line replaces that footer — the prompt box itself (see
# :func:`_prompt_box_shown`). A newer Claude Code that redraws these shows up as
# a ``repl-ready`` failure that prints the terminal's last lines.
TRUST_MARKERS: tuple[str, ...] = ("Yes, I trust this folder", "Quick safety check")
REPL_READY_MARKERS: tuple[str, ...] = ("? for shortcuts",)
# The selection cursor Claude Code draws beside the highlighted option.
MENU_CURSOR = "❯"
# Arrow presses allowed while steering the cursor onto the trust option.
_TRUST_NAV_MAX = 4

POLL_MS = 200
RELAYOUT_TIMEOUT_MS = 2_000

ARTIFACTS_IFRAME = 'iframe.panel-iframe[data-panel-id="artifacts"]'
PLOT_IFRAME = "iframe.preview-iframe-light"
ARIEL_IFRAME = 'iframe[data-panel-id="ariel"]'


class ProbeFailed(Exception):
    """A probe gave up; ``step`` names the storyboard step that failed."""

    def __init__(self, step: str, detail: str) -> None:
        super().__init__(f"{step}: {detail}")
        self.step = step
        self.detail = detail


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------

_FOCUS_JS = (
    "() => { const a = document.activeElement;"
    " return !!a && a.classList.contains('xterm-helper-textarea'); }"
)

# xterm's DOM renderer keeps one child element per row inside .xterm-rows;
# joining the rows with newlines keeps line boundaries that the flat
# textContent of the container would lose.
_XTERM_TEXT_JS = (
    "() => { const r = document.querySelector('.xterm-rows');"
    " if (!r) return '';"
    " return Array.from(r.children).map((c) => c.textContent || '').join('\\n'); }"
)


def terminal_focused(page: Any) -> bool:
    """Return whether keyboard focus is on xterm's helper textarea."""
    return bool(page.evaluate(_FOCUS_JS))


def xterm_text(page: Any) -> str:
    """Return the visible terminal text, one line per xterm row.

    xterm's DOM renderer writes some spaces as U+00A0; they come back as plain
    spaces so a marker can spell them the ordinary way.
    """
    return (page.evaluate(_XTERM_TEXT_JS) or "").replace("\u00a0", " ")


def terminal_tail(page: Any) -> str:
    """The terminal's last 5 non-blank lines, or "" when the page cannot answer.

    For failure messages: what was on screen when a step gave up.
    """
    try:
        return last_lines(xterm_text(page))
    except Exception:
        return ""


def last_lines(text: str, n: int = 5) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def _prompt_box_shown(text: str) -> bool:
    """Whether the idle input box is drawn: a cursor line right under a rule.

    The trust dialog's menu cursor has no rule above it, so it never matches.
    """
    lines = [line.strip() for line in text.splitlines()]
    return any(
        line.startswith(MENU_CURSOR) and len(above) >= 10 and set(above) == {"─"}
        for above, line in zip(lines, lines[1:], strict=False)
    )


def _trust_key(text: str) -> str:
    """The key that moves the trust dialog toward accepting: an arrow or Enter.

    Enter only once the cursor sits on the "trust" option, since the option
    order varies between Claude Code versions and Enter on "No, exit" ends the
    session. A dialog drawn without a cursor gets Enter.
    """
    lines = text.splitlines()
    cursor = next((i for i, line in enumerate(lines) if MENU_CURSOR in line), None)
    yes = next((i for i, line in enumerate(lines) if TRUST_MARKERS[0] in line), None)
    if cursor is None or yes is None or cursor == yes:
        return "Enter"
    return "ArrowDown" if yes > cursor else "ArrowUp"


def await_repl_ready(page: Any, budget_s: float = 60) -> None:
    """Wait until the Claude Code REPL is idle, accepting the trust dialog once.

    The trust dialog may paint after polling starts, so it is looked for on
    every tick. The cursor is stepped onto the "trust" option one arrow per
    tick, and Enter is pressed once it is there, and never again.
    Raises :class:`ProbeFailed` with step ``repl-ready`` and the terminal's
    last 5 lines when no REPL marker shows up within *budget_s*.
    """
    ticks = max(1, math.ceil(budget_s * 1000 / POLL_MS))
    trusted = False
    nav_presses = 0
    text = ""
    for _ in range(ticks):
        text = xterm_text(page)
        if any(m in text for m in REPL_READY_MARKERS) or _prompt_box_shown(text):
            return
        if not trusted and any(m in text for m in TRUST_MARKERS):
            key = _trust_key(text)
            if key == "Enter":
                page.keyboard.press("Enter")
                trusted = True
            elif nav_presses < _TRUST_NAV_MAX:
                page.keyboard.press(key)
                nav_presses += 1
        page.wait_for_timeout(POLL_MS)
    raise ProbeFailed(
        "repl-ready",
        f"no REPL marker within {budget_s:g} s; last lines:\n{last_lines(text)}",
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _child_frame(parent: Any, selector: str) -> Any | None:
    handle = parent.locator(selector).first.element_handle()
    if handle is None:
        return None
    return handle.content_frame()


def plot_frame(page: Any) -> Any:
    """Return the Frame holding the plot preview inside the artifacts panel."""
    panel = _child_frame(page, ARTIFACTS_IFRAME)
    if panel is None:
        raise ProbeFailed("rotate", "artifacts panel iframe not found")
    plot = _child_frame(panel, PLOT_IFRAME)
    if plot is None:
        raise ProbeFailed("rotate", "plot preview iframe not found")
    return plot


# ---------------------------------------------------------------------------
# Gallery cards and their previews (inside the artifacts panel)
# ---------------------------------------------------------------------------

_CARD_SHOWN_JS = '(id) => !!document.querySelector(`[data-id="${CSS.escape(id)}"]`)'

# The preview pane shows the artifact whose file path names ``id`` (every
# stored filename starts with its id), and its viewport has content: an
# iframe whose document has loaded a body, an image that decoded, a
# time-series viewer whose Plotly chart has drawn its traces, or a container
# the renderer has filled.
_PREVIEW_SHOWS_JS = """(id) => {
  const content = document.getElementById('preview-content');
  if (!content || content.classList.contains('hidden')) return false;
  const path = content.querySelector('.preview-path-text');
  if (!path || !path.textContent.includes(id)) return false;
  const shown = content.querySelector('.preview-viewport > *');
  if (!shown) return false;
  if (shown.tagName === 'IFRAME') {
    try {
      const doc = shown.contentDocument;
      return !!doc && doc.readyState === 'complete' && !!doc.body
        && doc.body.childElementCount > 0;
    } catch (e) { return false; }
  }
  if (shown.tagName === 'IMG') return shown.complete && shown.naturalWidth > 0;
  if (shown.classList.contains('preview-download')) return true;
  if (shown.classList.contains('ts-viewport-container')) {
    if (shown.querySelector('.ts-loading')) return false;
    const chart = shown.querySelector('[data-ts-chart]');
    if (chart) return !!chart._fullLayout && !!chart.querySelector('.scatterlayer .trace, canvas');
  }
  return shown.childElementCount > 0 || shown.textContent.trim().length > 0;
}"""

_PREVIEW_STATE_JS = """() => {
  const content = document.getElementById('preview-content');
  if (!content || content.classList.contains('hidden')) return 'preview: empty';
  const title = content.querySelector('.preview-header-title');
  const shown = content.querySelector('.preview-viewport > *');
  const what = shown ? (shown.tagName.toLowerCase() + '.' + shown.className) : 'nothing';
  return `preview: ${title ? title.textContent.trim() : '?'} showing ${what}`;
}"""


def _panel(page: Any) -> Any | None:
    try:
        return _child_frame(page, ARTIFACTS_IFRAME)
    except Exception:
        return None


def card_shown(page: Any, artifact_id: str) -> bool:
    """Whether the gallery lists a card for ``artifact_id``."""
    panel = _panel(page)
    return bool(panel is not None and panel.evaluate(_CARD_SHOWN_JS, artifact_id))


def preview_shows(page: Any, artifact_id: str) -> bool:
    """Whether the preview pane has rendered ``artifact_id``."""
    panel = _panel(page)
    return bool(panel is not None and panel.evaluate(_PREVIEW_SHOWS_JS, artifact_id))


def preview_state(page: Any) -> str:
    """One line on what the preview pane shows, for failure messages."""
    panel = _panel(page)
    if panel is None:
        return "preview: artifacts panel not found"
    try:
        return str(panel.evaluate(_PREVIEW_STATE_JS))
    except Exception as exc:
        return f"preview: unreadable ({exc})"


# The live WebGL camera first: it is what the viewer sees. Plotly copies it
# into the layout only when the mouse is released over the canvas, so a drag
# that ends over the colorbar turns the scene but leaves the layout behind.
_CAMERA_EYE_JS = (
    "() => { const gd = document.querySelector('.js-plotly-plot');"
    " const s = gd && gd._fullLayout && gd._fullLayout.scene;"
    " const live = s && s._scene && s._scene.getCamera && s._scene.getCamera();"
    " const e = (live && live.eye) || (s && s.camera && s.camera.eye);"
    " return e ? {x: e.x, y: e.y, z: e.z} : null; }"
)

_ARM_RELAYOUT_JS = (
    "() => { const gd = document.querySelector('.js-plotly-plot');"
    " if (!gd || !gd.on) { window.__videoRelayout = null; return false; }"
    " const sub = (gd.once || gd.on).bind(gd);"
    " window.__videoRelayout = new Promise((res) => sub('plotly_relayout', () => res(true)));"
    " return true; }"
)

_AWAIT_RELAYOUT_JS = (
    "(ms) => { const p = window.__videoRelayout; window.__videoRelayout = null;"
    " if (!p) return false;"
    " return Promise.race([p, new Promise((res) => setTimeout(() => res(false), ms))]); }"
)


def camera_azimuth(frame: Any) -> float:
    """Return the 3D camera azimuth, ``atan2(eye.y, eye.x)``, in degrees."""
    eye = frame.evaluate(_CAMERA_EYE_JS)
    if not eye:
        raise ProbeFailed("rotate", "no 3D scene camera on the plot")
    return math.degrees(math.atan2(eye["y"], eye["x"]))


def azimuth_change(before: float, after: float) -> float:
    """Return the absolute angle between two azimuths in degrees, in ``[0, 180]``."""
    delta = (after - before) % 360.0
    return min(delta, 360.0 - delta)


def arm_relayout(frame: Any) -> bool:
    """Register a one-shot ``plotly_relayout`` listener; call before the drag."""
    return bool(frame.evaluate(_ARM_RELAYOUT_JS))


def await_relayout(frame: Any, timeout_ms: int = RELAYOUT_TIMEOUT_MS) -> bool:
    """Wait for the armed relayout, or *timeout_ms*; True when it fired."""
    return bool(frame.evaluate(_AWAIT_RELAYOUT_JS, timeout_ms))


# ---------------------------------------------------------------------------
# Logbook draft
# ---------------------------------------------------------------------------

# The image is drawn to a canvas and its pixels reduced to grey levels; a
# blank (single-colour) PNG has exactly one. The attachment is served from the
# same origin, so the canvas is not tainted; if it were, getImageData throws
# and the draft counts as not ready.
_DRAFT_JS = """() => {
  const banner = document.querySelector('#draft-banner[data-draft-id]');
  const img = document.querySelector('#file-preview img');
  const out = {banner: !!banner, image: !!img && img.complete && img.naturalWidth > 0,
               greyLevels: 0};
  if (!out.image) return out;
  try {
    const w = Math.min(img.naturalWidth, 256), h = Math.min(img.naturalHeight, 256);
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
    const d = ctx.getImageData(0, 0, w, h).data;
    const seen = new Set();
    for (let i = 0; i < d.length; i += 4) {
      seen.add(Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]));
      if (seen.size > 1) break;
    }
    out.greyLevels = seen.size;
  } catch (e) {
    out.greyLevels = 0;
  }
  return out;
}"""


def draft_ready(page: Any) -> bool:
    """Return whether the ARIEL panel shows a saved draft with a non-blank image."""
    frame = _child_frame(page, ARIEL_IFRAME)
    if frame is None:
        return False
    state = frame.evaluate(_DRAFT_JS) or {}
    return bool(state.get("banner") and state.get("image") and state.get("greyLevels", 0) > 1)


# The attachment preview sits below the draft form's fold; the closing hold
# scrolls it to the middle of the ARIEL panel so the attached plot is on screen.
_REVEAL_ATTACHMENT_JS = """() => {
  const img = document.querySelector('#file-preview img') || document.querySelector('#file-preview');
  if (!img) return false;
  img.scrollIntoView({block: 'center', behavior: 'smooth'});
  return true;
}"""


def reveal_attachment(page: Any) -> bool:
    """Scroll the draft's attached image into view; False when there is none."""
    frame = _child_frame(page, ARIEL_IFRAME)
    return bool(frame is not None and frame.evaluate(_REVEAL_ATTACHMENT_JS))
