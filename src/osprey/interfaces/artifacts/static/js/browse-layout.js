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

import { clampSize as clamp, initSplitter } from '/design-system/js/splitter.js';

const ORIENT_KEY = 'osprey-artifacts-browse-orient';
const WIDTH_KEY = 'osprey-artifacts-browse-sidebar-width';
const HEIGHT_KEY = 'osprey-artifacts-browse-sidebar-height';
const MIN_WIDTH = 200;
const MAX_WIDTH = 560;
const MIN_HEIGHT = 120;
const MAX_HEIGHT = 480;
const KEY_STEP = 16;

/** Below this the panel is too narrow for a side-by-side split; gallery.css
 *  forces the stacked layout here regardless of the persisted choice. */
const STACK_QUERY = '(max-width: 560px)';

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

/** @returns {boolean} whether the viewport is forcing the stacked layout */
function isForcedStacked() {
  return typeof window.matchMedia === 'function' && window.matchMedia(STACK_QUERY).matches;
}

/**
 * The orientation actually on screen, which is not always the persisted one.
 *
 * `data-browse-orient` records the operator's choice, but gallery.css also
 * forces a stacked layout below 560px through a media query the attribute
 * never learns about. Gating the splitters on the attribute alone would leave
 * the narrow viewport with the x-handle hidden by CSS and the y-handle parked
 * by JS — a panel with no working divider at all.
 *
 * @returns {Orient}
 */
export function effectiveOrient() {
  return getOrient() === 'column' || isForcedStacked() ? 'column' : 'row';
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
 * @param {HTMLElement|null} els.handle    splitter for the side-by-side split
 * @param {HTMLElement|null} [els.handleY] splitter for the stacked split
 * @param {HTMLElement|null} els.sidebar   browser pane (left / top)
 * @param {HTMLElement|null} els.toggle    orientation toggle button
 * @returns {void}
 */
export function initBrowseLayout({ handle, handleY = null, sidebar, toggle }) {
  // One instance per axis. A splitter binds its axis at construction — the
  // geometry it writes, the cursor, and the arrow-key mapping all follow from
  // it — so a view that flips axis needs two, each parked by `isEnabled` when
  // the other is showing. Re-read per interaction, so a flip parks a splitter
  // without detaching listeners that would need re-attaching on the way back.
  const splitter = initSplitter({
    handle,
    pane: sidebar,
    storageKey: WIDTH_KEY,
    min: MIN_WIDTH,
    max: MAX_WIDTH,
    step: KEY_STEP,
    collapsedSize: 0,
    isEnabled: () => effectiveOrient() === 'row',
  });

  const splitterY = initSplitter({
    handle: handleY,
    pane: sidebar,
    storageKey: HEIGHT_KEY,
    axis: 'y',
    min: MIN_HEIGHT,
    max: MAX_HEIGHT,
    step: KEY_STEP,
    collapsedSize: 0,
    isEnabled: () => effectiveOrient() === 'column',
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

  /**
   * Wake the splitter matching the effective orientation and park the other.
   *
   * Clearing the parked one matters: an inline flex-basis left over from the
   * x-axis would keep constraining the pane after the layout flips to y.
   *
   * @param {Orient} orient the persisted choice to record
   */
  const applyOrient = (orient) => {
    document.documentElement.dataset.browseOrient = orient;
    if (effectiveOrient() === 'row') {
      splitterY.clearSize();
      splitter.restoreSize();
    } else {
      splitter.clearSize();
      splitterY.restoreSize();
    }
    syncToggle();
  };

  // Restore: index.html ships data-browse-orient="row" (the default); a
  // persisted "column" choice re-applies here, before first render work.
  applyOrient(normalizeOrient(readStored(ORIENT_KEY)));

  // The forced-stacked threshold is a viewport fact, not a stored choice, so a
  // resize across it has to re-park the splitters without touching what the
  // operator picked.
  if (typeof window.matchMedia === 'function') {
    window.matchMedia(STACK_QUERY).addEventListener('change', () => applyOrient(getOrient()));
  }

  if (toggle) {
    toggle.addEventListener('click', () => {
      const next = getOrient() === 'column' ? 'row' : 'column';
      persist(ORIENT_KEY, next);
      applyOrient(next);
    });
  }
}
