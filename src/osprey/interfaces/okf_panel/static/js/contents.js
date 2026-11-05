// @ts-check
/*
 * OKF Knowledge Panel — the contents list (the panel sidebar) and when it
 * shows.
 *
 * The list is the design system's panel sidebar (panel-sidebar.js): its
 * toggle is the toolbar's `panel-left` button, its divider the shared
 * splitter, and in a narrow tile it opens as an overlay over the reader. This
 * module adds only the Knowledge rule.
 *
 * The structure overview is itself a table of contents, so there the list
 * starts closed and the overview has the reader's full width. Opening a page
 * shows the list, unless the operator hid it while reading a page: that
 * choice is remembered across pages and reloads, with the width. A hide or
 * show on the overview is not remembered; the next overview starts closed.
 *
 * Search results are drawn in the list, so a search shows it; clearing the
 * search hands the list back to the view's rule. The rule is applied when the
 * view changes (overview to page or back) and when a search changes, never on
 * a page-to-page step, so an operator's hide or show holds while they read
 * on. A row picked in the narrow overlay closes it, so the page shows.
 *
 * The stored record keeps its key, and its `{size, collapsed}` shape is the
 * one panel-sidebar.js reads, so a width and a hide stored before this module
 * used the shared sidebar still hold.
 */

import { initPanelSidebar } from '/design-system/js/panel-sidebar.js';

const STORAGE_KEY = 'osprey-okf-sidebar-width';

/** @typedef {'browse' | 'page'} ReaderView */

/** @type {HTMLElement | null} */
let sidebarEl = null;
/** @type {import('/design-system/js/panel-sidebar.js').PanelSidebarApi | null} */
let sidebar = null;
/** @type {ReaderView} */
let view = 'browse';
let searching = false;

/**
 * Wire the list, its toggle and its divider. Call once at boot, before the
 * first view is shown; the reader starts on the overview. No-ops when the
 * list is absent (a stripped-down embed).
 */
export function initContents() {
  sidebarEl = document.getElementById('sidebar');
  if (!sidebarEl) return;
  sidebar = initPanelSidebar({
    sidebar: sidebarEl,
    handle: document.getElementById('sidebar-resizer'),
    toggle: document.getElementById('contents-toggle'),
    storageKey: STORAGE_KEY,
    defaultOpen: () => view === 'page',
  });
}

/**
 * The reader is about to show `next`. Applies the view's rule when the view
 * changes; a page-to-page step leaves the list as it is.
 * @param {ReaderView} next
 */
export function showContentsFor(next) {
  if (next === view) return;
  view = next;
  if (searching) sidebar?.open();
  else sidebar?.applyDefault();
}

/**
 * A search was run (`active`) or cleared. Every search shows the list, which
 * is where its results are; clearing it applies the view's rule again.
 * @param {boolean} active
 */
export function setContentsSearching(active) {
  if (!active && !searching) return;
  searching = active;
  if (active) sidebar?.open();
  else sidebar?.applyDefault();
}

/** A row of the list was picked: the narrow overlay makes way for the page. */
export function contentsPicked() {
  if (sidebarEl?.hasAttribute('data-overlay')) sidebar?.close();
}

/** The tile ⋯ entry "Reset contents width": back to the default width. */
export function resetContentsWidth() {
  sidebar?.resetWidth();
}
