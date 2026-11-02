// @ts-check
/*
 * OKF Knowledge Panel — sidebar resizer.
 *
 * A thin binding over the design system's shared splitter
 * (/design-system/js/splitter.js): this file owns only what is specific to
 * this panel — which elements are the pane and the handle, the usable width
 * range, and the stored record. Drag mechanics, clamping and the
 * ARIA-separator keyboard nudge all live in the shared module, so the OKF
 * split behaves identically to the artifact gallery's and the PLAN panel's.
 * The grip-pill visual is likewise shared, as `.osprey-splitter` in the design
 * system's base.css.
 *
 * The splitter's collapse is also whether the sidebar shows at all, and that
 * depends on the view (contents.js): a hide the panel makes on its own must
 * not overwrite the hide or show an operator chose. So the splitter persists
 * nothing (`storageKey: null`); it reports each settled change to the caller,
 * which writes the record here with {@link writeSidebarRecord}.
 */

import { clampSize as clamp, initSplitter, parseStoredState } from '/design-system/js/splitter.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';

const STORAGE_KEY = 'osprey-okf-sidebar-width';
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

/** @typedef {{size: number | null, collapsed: boolean}} SidebarRecord */

/**
 * The stored sidebar record: its width, and whether the operator hid it on a
 * page. A bare integer from before the splitter had a collapse still reads as
 * a width.
 * @returns {SidebarRecord}
 */
export function readSidebarRecord() {
  let raw = null;
  try {
    raw = localStorage.getItem(scopedStorageKey(STORAGE_KEY));
  } catch {
    /* storage blocked — the stylesheet's width, shown on pages */
  }
  return parseStoredState(raw, false);
}

/**
 * @param {SidebarRecord} record
 */
export function writeSidebarRecord(record) {
  try {
    localStorage.setItem(scopedStorageKey(STORAGE_KEY), JSON.stringify(record));
  } catch {
    /* storage blocked — the choice holds for this page lifetime */
  }
}

/**
 * Wire the sidebar resizer. The splitter starts collapsed, as index.html
 * starts the sidebar hidden; it opens at the stored width, else at the
 * stylesheet's. No-ops when the sidebar is absent (a stripped-down embed).
 * @param {{onCommit: (size: number, collapsed: boolean) => void}} opts
 *   called after every settled change: a drag, a nudge, a collapse or expand
 * @returns {import('/design-system/js/splitter.js').SplitterApi}
 */
export function initSidebarResize({ onCommit }) {
  const pane = document.getElementById('sidebar');
  const cssWidth = pane ? parseFloat(getComputedStyle(pane).flexBasis) : NaN;
  return initSplitter({
    handle: document.getElementById('sidebar-resizer'),
    pane,
    storageKey: null,
    min: MIN_WIDTH,
    max: MAX_WIDTH,
    step: KEY_STEP,
    // Hidden is hidden outright: there is no header to keep visible, and the
    // reader is the pane an operator wants all of.
    collapsedSize: 0,
    initialSize: readSidebarRecord().size ?? (Number.isFinite(cssWidth) ? cssWidth : null),
    initialCollapsed: true,
    onCommit,
  });
}
