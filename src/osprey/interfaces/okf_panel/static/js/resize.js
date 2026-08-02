// @ts-check
/*
 * OKF Knowledge Panel — sidebar resizer.
 *
 * A thin binding over the design system's shared splitter
 * (/design-system/js/splitter.js): this file owns only what is specific to
 * this panel — which elements are the pane and the handle, the usable width
 * range, and the storage key. Drag mechanics, clamping, persistence and the
 * ARIA-separator keyboard nudge all live in the shared module, so the OKF
 * split behaves identically to the artifact gallery's and the PLAN panel's.
 * The grip-pill visual is likewise shared, as `.osprey-splitter` in the design
 * system's base.css.
 */

import { clampSize as clamp, initSplitter } from "/design-system/js/splitter.js";

const STORAGE_KEY = "osprey-okf-sidebar-width";
const MIN_WIDTH = 180;
const MAX_WIDTH = 560;
const KEY_STEP = 16;

/**
 * Clamp a candidate sidebar width to this panel's usable range.
 * @param {number} width
 * @returns {number}
 */
export function clampWidth(width) {
  return clamp(width, MIN_WIDTH, MAX_WIDTH);
}

/**
 * Wire the sidebar resizer and restore the persisted split. No-ops when
 * either element is absent (e.g. a stripped-down embed). Safe to call once
 * at boot.
 */
export function initSidebarResize() {
  initSplitter({
    handle: document.getElementById("sidebar-resizer"),
    pane: document.getElementById("sidebar"),
    storageKey: STORAGE_KEY,
    min: MIN_WIDTH,
    max: MAX_WIDTH,
    step: KEY_STEP,
    // Double-click hides the sidebar outright: there is no header to keep
    // visible, and the reader is the pane an operator wants all of.
    collapsedSize: 0,
  }).restoreSize();
}
