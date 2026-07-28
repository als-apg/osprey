// @ts-check
/**
 * OSPREY Artifact Gallery — browse-view layout: orientation + splitter.
 *
 * The browse view is a browser card plus an artifact detail card in one
 * flex container. This module owns the axis of that split and the divider
 * between the cards:
 *
 * - Orientation: side-by-side (browser left, artifact right — the default)
 *   or stacked (browser band on top). Stamped as `data-browse-orient` on
 *   <html> so CSS keys every delta off one attribute, persisted per origin,
 *   flipped by the header toggle button.
 * - Splitter: in side-by-side mode the divider is a live pointer-driven
 *   splitter — same grip-pill idiom, clamping, persistence and ARIA
 *   separator keyboard support as the OKF panel's sidebar resizer. Stacked
 *   mode has a fixed-height band (no splitter), matching the OKF panel's
 *   stacked responsive layout.
 */

const ORIENT_KEY = "osprey-artifacts-browse-orient";
const WIDTH_KEY = "osprey-artifacts-browse-sidebar-width";
const MIN_WIDTH = 200;
const MAX_WIDTH = 560;
const KEY_STEP = 16;

/** @typedef {"row"|"column"} Orient */

/**
 * Normalize a stored/candidate orientation; anything but "column" is "row".
 * @param {string|null|undefined} value
 * @returns {Orient}
 */
export function normalizeOrient(value) {
  return value === "column" ? "column" : "row";
}

/**
 * Clamp a candidate sidebar width to the usable range.
 * @param {number} width
 * @returns {number}
 */
export function clampWidth(width) {
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, Math.round(width)));
}

/** @returns {Orient} the current applied orientation */
export function getOrient() {
  return normalizeOrient(document.documentElement.dataset.browseOrient);
}

/**
 * Read a persisted value; storage may be blocked (sandboxed iframe).
 * @param {string} key
 * @returns {string|null}
 */
function readStored(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/** @param {string} key @param {string} value */
function persist(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* storage blocked — the choice still holds for this page lifetime */
  }
}

/**
 * Wire the browse-view layout: restore persisted orientation and split
 * width, activate the splitter, and bind the orientation toggle. No-ops
 * for whichever elements are absent. Safe to call once at boot.
 *
 * @param {object} els
 * @param {HTMLElement|null} els.handle    splitter between the cards
 * @param {HTMLElement|null} els.sidebar   browser card (left / top)
 * @param {HTMLElement|null} els.toggle    orientation toggle button
 * @returns {void}
 */
export function initBrowseLayout({ handle, sidebar, toggle }) {
  /** @param {number} width @returns {number} the applied (clamped) width */
  const applyWidth = (width) => {
    const px = clampWidth(width);
    if (sidebar) {
      sidebar.style.flexBasis = `${px}px`;
      sidebar.style.maxWidth = `${px}px`;
    }
    return px;
  };

  /** Inline sizes are row-mode state; the stacked band is styled by CSS alone. */
  const clearWidth = () => {
    if (sidebar) {
      sidebar.style.flexBasis = "";
      sidebar.style.maxWidth = "";
    }
  };

  const syncToggle = () => {
    if (!toggle) return;
    const stacked = getOrient() === "column";
    const label = stacked ? "Switch to side-by-side layout" : "Switch to stacked layout";
    toggle.setAttribute("aria-label", label);
    toggle.title = label;
  };

  /** @param {Orient} orient */
  const applyOrient = (orient) => {
    document.documentElement.dataset.browseOrient = orient;
    if (orient === "row") {
      const stored = parseInt(readStored(WIDTH_KEY) ?? "", 10);
      if (Number.isFinite(stored)) applyWidth(stored);
    } else {
      clearWidth();
    }
    syncToggle();
  };

  // Restore: index.html ships data-browse-orient="row" (the default); a
  // persisted "column" choice re-applies here, before first render work.
  applyOrient(normalizeOrient(readStored(ORIENT_KEY)));

  if (toggle) {
    toggle.addEventListener("click", () => {
      const next = getOrient() === "column" ? "row" : "column";
      persist(ORIENT_KEY, next);
      applyOrient(next);
    });
  }

  if (!handle || !sidebar) return;

  handle.addEventListener("pointerdown", (e) => {
    if (getOrient() !== "row") return;
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    handle.classList.add("dragging");
    const startX = e.clientX;
    const startWidth = sidebar.getBoundingClientRect().width;
    let width = startWidth;

    const move = (/** @type {PointerEvent} */ ev) => {
      width = applyWidth(startWidth + (ev.clientX - startX));
    };
    const end = () => {
      handle.classList.remove("dragging");
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", end);
      handle.removeEventListener("pointercancel", end);
      persist(WIDTH_KEY, String(width));
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
  });

  handle.addEventListener("keydown", (e) => {
    if (getOrient() !== "row") return;
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const current = sidebar.getBoundingClientRect().width;
    const next = applyWidth(current + (e.key === "ArrowRight" ? KEY_STEP : -KEY_STEP));
    persist(WIDTH_KEY, String(next));
  });
}
