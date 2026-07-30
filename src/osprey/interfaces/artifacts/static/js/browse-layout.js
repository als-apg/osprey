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
 *   flipped by the header toggle button. This axis is the gallery's own
 *   feature — no other panel has it.
 * - Splitter: the divider itself is the design system's shared splitter
 *   (/design-system/js/splitter.js, `.osprey-splitter` in base.css), the same
 *   drag/clamp/persist/keyboard behaviour the OKF and PLAN panels get. It is
 *   gated on the orientation: stacked mode is a fixed-height band with no
 *   split to drag, so the handle goes inert rather than being torn down.
 */

import { clampWidth as clamp, initSplitter } from '/design-system/js/splitter.js';

const ORIENT_KEY = 'osprey-artifacts-browse-orient';
const WIDTH_KEY = 'osprey-artifacts-browse-sidebar-width';
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
  return value === 'column' ? 'column' : 'row';
}

/**
 * Clamp a candidate sidebar width to this view's usable range.
 * @param {number} width
 * @returns {number}
 */
export function clampWidth(width) {
  return clamp(width, MIN_WIDTH, MAX_WIDTH);
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
  const splitter = initSplitter({
    handle,
    pane: sidebar,
    storageKey: WIDTH_KEY,
    min: MIN_WIDTH,
    max: MAX_WIDTH,
    step: KEY_STEP,
    // Re-read per interaction, so flipping the axis parks the splitter
    // without detaching listeners that would need re-attaching on the way back.
    isEnabled: () => getOrient() === 'row',
  });

  const syncToggle = () => {
    if (!toggle) return;
    const stacked = getOrient() === 'column';
    const label = stacked ? 'Switch to side-by-side layout' : 'Switch to stacked layout';
    toggle.setAttribute('aria-label', label);
    toggle.title = label;
    // Menu-item form of the toggle carries a visible label naming the
    // layout a click switches TO; plain icon-button hosts have no span.
    const labelEl = toggle.querySelector('.orient-label');
    if (labelEl) labelEl.textContent = stacked ? 'Side-by-side layout' : 'Stacked layout';
  };

  /** @param {Orient} orient */
  const applyOrient = (orient) => {
    document.documentElement.dataset.browseOrient = orient;
    // Inline sizes are row-mode state; the stacked band is styled by CSS alone.
    if (orient === 'row') splitter.restoreWidth();
    else splitter.clearWidth();
    syncToggle();
  };

  // Restore: index.html ships data-browse-orient="row" (the default); a
  // persisted "column" choice re-applies here, before first render work.
  applyOrient(normalizeOrient(readStored(ORIENT_KEY)));

  if (toggle) {
    toggle.addEventListener('click', () => {
      const next = getOrient() === 'column' ? 'row' : 'column';
      persist(ORIENT_KEY, next);
      applyOrient(next);
    });
  }
}
