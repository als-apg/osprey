// @ts-check
/*
 * OKF Knowledge Panel — when the contents list (the sidebar) shows.
 *
 * The structure overview is itself a table of contents, so there the list
 * starts hidden and the overview has the reader's full width. Opening a page
 * shows the list, unless the operator hid it while reading a page: that
 * choice is remembered across pages and reloads, with the width. A hide or
 * show on the overview is not remembered; the next overview starts hidden.
 *
 * Search results are drawn in the list, so a search shows it; clearing the
 * search hands the list back to the view's rule. A narrow tile stacks the list
 * above the page, where it pushes the page down, so there it starts hidden on
 * every view and a show is not remembered.
 *
 * The rule is applied when the view changes (overview to page or back), when
 * a search changes, and when the tile crosses the narrow width — never on a
 * page-to-page step, so an operator's hide or show holds while they read on.
 *
 * One state, two mirrors: the shared splitter's collapse is the state
 * (resize.js), and after every change `#app[data-contents]` (the stylesheet's
 * hook) and the toggle button follow it. index.html stamps "closed" so the
 * first paint never shows a list the boot is about to hide.
 */

import { icon } from '/design-system/js/icons.js';
import { initSidebarResize, readSidebarRecord, writeSidebarRecord } from './resize.js';

/** style.css stacks the list above the reader at this width. */
const NARROW_QUERY = '(max-width: 720px)';

/** @typedef {'browse' | 'page'} ReaderView */

/** @type {HTMLElement | null} */
let appEl = null;
/** @type {HTMLElement | null} */
let toggleEl = null;
/** @type {import('/design-system/js/splitter.js').SplitterApi | null} */
let split = null;
/** @type {MediaQueryList | null} */
let narrowList = null;
/** @type {import('./resize.js').SidebarRecord} */
let record = { size: null, collapsed: false };
/** @type {ReaderView} */
let view = 'browse';
let searching = false;
/** True while the panel itself hides or shows the list: nothing is stored. */
let automatic = false;

/**
 * Wire the toggle and the sidebar's splitter. Call once at boot, before the
 * first view is shown; the reader starts on the overview.
 */
export function initContents() {
  appEl = document.getElementById('app');
  toggleEl = document.getElementById('contents-toggle');
  record = readSidebarRecord();
  split = initSidebarResize({ onCommit });
  narrowList = window.matchMedia?.(NARROW_QUERY) ?? null;
  narrowList?.addEventListener('change', apply);
  if (toggleEl) {
    toggleEl.replaceChildren(icon('panel-left'));
    toggleEl.addEventListener('click', () => split?.toggleCollapse());
  }
  reflect();
}

/**
 * The reader is about to show `next`. Applies the view's rule when the view
 * changes; a page-to-page step leaves the list as it is.
 * @param {ReaderView} next
 */
export function showContentsFor(next) {
  if (next === view) return;
  view = next;
  apply();
}

/**
 * A search was run (`active`) or cleared. Every search shows the list, which
 * is where its results are; clearing it applies the view's rule again.
 * @param {boolean} active
 */
export function setContentsSearching(active) {
  if (!active && !searching) return;
  searching = active;
  apply();
}

/** @returns {boolean} whether the current view, on its own, wants the list. */
function wanted() {
  if (searching) return true;
  if (view === 'browse' || narrowList?.matches) return false;
  return !record.collapsed;
}

/** Bring the list to what the view wants; nothing it does is stored. */
function apply() {
  if (!split) return;
  automatic = true;
  try {
    if (wanted()) {
      if (split.isCollapsed()) split.expand();
    } else if (!split.isCollapsed()) {
      split.collapse();
    }
  } finally {
    automatic = false;
  }
  reflect();
}

/**
 * The splitter settled a change. One the operator made — the toggle, the
 * divider's double-click or Enter, a drag or an arrow-key nudge — stores the
 * width, and on a page in a wide tile whether the list is hidden.
 * @param {number} size
 * @param {boolean} collapsed
 */
function onCommit(size, collapsed) {
  reflect();
  if (automatic) return;
  record.size = size;
  if (view === 'page' && !narrowList?.matches) record.collapsed = collapsed;
  writeSidebarRecord(record);
}

/** Mirror the splitter's state onto `#app` and the toggle. */
function reflect() {
  const open = !(split?.isCollapsed() ?? true);
  appEl?.setAttribute('data-contents', open ? 'open' : 'closed');
  if (!toggleEl) return;
  const action = open ? 'Hide contents' : 'Show contents';
  toggleEl.setAttribute('aria-expanded', String(open));
  toggleEl.setAttribute('aria-label', action);
  toggleEl.title = action;
}
