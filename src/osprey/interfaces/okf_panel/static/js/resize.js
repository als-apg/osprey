// @ts-check
/*
 * OKF Knowledge Panel — sidebar resizer.
 *
 * A pointer-driven splitter between the concept-tree sidebar and the reading
 * pane, echoing the web-terminal hub's dockview sash (same grip-pill visual,
 * same instant accent feedback — see style.css #sidebar-resizer). Width is
 * clamped, applied as the sidebar's flex-basis, and persisted per origin so
 * the chosen split survives reloads. Keyboard: the separator is focusable and
 * Arrow keys nudge the split, per the ARIA separator pattern.
 */

const STORAGE_KEY = "osprey-okf-sidebar-width";
const MIN_WIDTH = 180;
const MAX_WIDTH = 560;
const KEY_STEP = 16;

/**
 * Clamp a candidate sidebar width to the usable range.
 * @param {number} width
 * @returns {number}
 */
export function clampWidth(width) {
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, Math.round(width)));
}

/**
 * Wire the sidebar resizer. No-ops when either element is absent (e.g. a
 * stripped-down embed). Safe to call once at boot.
 */
export function initSidebarResize() {
  const resizer = document.getElementById("sidebar-resizer");
  const sidebar = document.getElementById("sidebar");
  if (!resizer || !sidebar) return;

  /** @param {number} width @returns {number} the applied (clamped) width */
  const apply = (width) => {
    const px = clampWidth(width);
    sidebar.style.flexBasis = `${px}px`;
    sidebar.style.maxWidth = `${px}px`;
    return px;
  };

  const persist = (/** @type {number} */ width) => {
    try {
      localStorage.setItem(STORAGE_KEY, String(width));
    } catch {
      /* storage blocked — the split still holds for this page lifetime */
    }
  };

  let stored = NaN;
  try {
    stored = parseInt(localStorage.getItem(STORAGE_KEY) ?? "", 10);
  } catch {
    /* storage blocked — keep the stylesheet default */
  }
  if (Number.isFinite(stored)) apply(stored);

  resizer.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    resizer.setPointerCapture(e.pointerId);
    resizer.classList.add("dragging");
    const startX = e.clientX;
    const startWidth = sidebar.getBoundingClientRect().width;
    let width = startWidth;

    const move = (/** @type {PointerEvent} */ ev) => {
      width = apply(startWidth + (ev.clientX - startX));
    };
    const end = () => {
      resizer.classList.remove("dragging");
      resizer.removeEventListener("pointermove", move);
      resizer.removeEventListener("pointerup", end);
      resizer.removeEventListener("pointercancel", end);
      persist(width);
    };
    resizer.addEventListener("pointermove", move);
    resizer.addEventListener("pointerup", end);
    resizer.addEventListener("pointercancel", end);
  });

  resizer.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const current = sidebar.getBoundingClientRect().width;
    const next = apply(current + (e.key === "ArrowRight" ? KEY_STEP : -KEY_STEP));
    persist(next);
  });
}
