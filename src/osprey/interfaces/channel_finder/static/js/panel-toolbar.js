// @ts-check
/**
 * OSPREY Channel Finder — the panel toolbar.
 *
 * The one 40 px row at the top of the panel, the same embedded and
 * standalone (DESIGN.md §12.3; the markup is in index.html):
 *
 *   [sidebar toggle] [Explore | Feedback] …… [result count] [search] [display menu] [⋯]
 *
 * The view switch belongs to the shell: a press sets the hash and app.js
 * routes on it; the pressed button follows `state.activeView`. The rest
 * belongs to the graph finder, the one view with a search, filters and
 * options. While it is mounted it claims the search field and the ⋯
 * ({@link claimToolbar}) and writes the result count ({@link setResultCount});
 * while its filters are drawn they drive the sidebar toggle
 * ({@link attachSidebar}). In every other view those controls keep their
 * slots, disabled, with a tooltip that says why, so a change of view moves
 * nothing.
 *
 * The ⋯ exists only on a page the hub does not frame (standalone, or popped
 * out): embedded, the tile's own ⋯ lists the same entries under "This panel"
 * (graph-finder-verbs.js publishMenu), and a tile has one ⋯. Both read one
 * entry list and run one handler.
 */

import { openContextMenu } from '/design-system/js/context-menu.js';
import { isEmbedded } from '/design-system/js/frame-params.js';
import { icon } from '/design-system/js/icons.js';
import { initPanelSidebar } from '/design-system/js/panel-sidebar.js';
import { writeTitle } from '/design-system/js/tooltip.js';
import { state } from './state.js';

/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */
/** @typedef {import('/design-system/js/header-contrib.js').HeaderMenuEntry} MenuEntry */
/** @typedef {import('/design-system/js/panel-sidebar.js').PanelSidebarApi} PanelSidebarApi */

/**
 * What a view hands the toolbar's ⋯: the entries, read each time it opens,
 * and the handler an entry runs.
 * @typedef {{entries: () => MenuEntry[], run: (id: string) => void}} PanelMenu
 */

/** The toolbar's elements, by id (index.html; the ⋯ is built here). */
export const TOOLBAR_IDS = Object.freeze({
  root: 'cf-panel',
  toggle: 'cf-sidebar-toggle',
  views: 'cf-views',
  end: 'cf-toolbar-end',
  count: 'cf-result-count',
  field: 'cf-search-field',
  search: 'cf-search',
  menu: 'cf-panel-menu',
});

const MENU_LABEL = 'Panel menu';
const SHOW_SIDEBAR = 'Show sidebar';

/** Why a control is disabled: under Feedback (when Explore has it), or anywhere else. */
const REASONS = Object.freeze({
  filters: ['Filters apply to Explore', 'No filters for this view'],
  search: ['Search applies to Explore', 'No search for this view'],
  menu: ['Options apply to Explore', 'No options for this view'],
});

/** The ⋯ of the view that claimed the toolbar, or null. @type {PanelMenu|null} */
let panelMenu = null;
/** Whether a view holds the search field. */
let searchClaimed = false;
/** Whether a drawn sidebar drives the toggle. */
let sidebarAttached = false;

/** @param {string} id @returns {HTMLElement|null} */
const byId = (id) => document.getElementById(id);

/**
 * The search field's input, which the graph finder reads and clears.
 * @returns {HTMLInputElement|null}
 */
export function searchInput() {
  return /** @type {HTMLInputElement|null} */ (byId(TOOLBAR_IDS.search));
}

/**
 * Why a control does not apply in the current view. Under Feedback the
 * controls point back to Explore, but only where Explore has them: the graph
 * paradigm is the one with filters, a search and options.
 * @param {keyof typeof REASONS} kind
 * @returns {string}
 */
function reason(kind) {
  const [feedback, other] = REASONS[kind];
  return state.activeView === 'feedback' && state.pipelineType === 'graph' ? feedback : other;
}

/** Bring every control in line with the view and with what is claimed. @returns {void} */
function refresh() {
  for (const btn of byId(TOOLBAR_IDS.views)?.querySelectorAll('button[data-view]') ?? []) {
    btn.setAttribute('aria-pressed', String(btn.getAttribute('data-view') === state.activeView));
  }

  const input = searchInput();
  if (input) input.disabled = !searchClaimed;
  const field = byId(TOOLBAR_IDS.field);
  if (field) writeTitle(field, searchClaimed ? '' : reason('search'));

  const menu = /** @type {HTMLButtonElement|null} */ (byId(TOOLBAR_IDS.menu));
  if (menu) {
    menu.disabled = panelMenu === null;
    writeTitle(menu, panelMenu ? MENU_LABEL : reason('menu'));
  }

  // A drawn sidebar keeps the toggle in step itself (panel-sidebar.js).
  const toggle = /** @type {HTMLButtonElement|null} */ (byId(TOOLBAR_IDS.toggle));
  if (toggle && !sidebarAttached) {
    toggle.disabled = true;
    toggle.setAttribute('aria-label', SHOW_SIDEBAR);
    toggle.removeAttribute('aria-expanded');
    toggle.removeAttribute('aria-controls');
    writeTitle(toggle, reason('filters'));
  }
}

/**
 * The ⋯ rows for a menu's entries: the same ids, labels and flags the hub's
 * "This panel" section draws, each running the view's handler.
 * @param {MenuEntry[]} entries
 * @param {(id: string) => void} run
 * @returns {MenuItem[]}
 */
export function menuRows(entries, run) {
  /** @type {MenuItem[]} */
  const rows = [];
  for (const entry of entries) {
    if (entry.separator && rows.length) rows.push({ divider: true });
    rows.push({
      label: entry.label,
      checked: entry.checked,
      disabled: entry.disabled,
      danger: entry.danger,
      run: () => run(entry.id),
    });
  }
  return rows;
}

/** @param {MouseEvent} event @returns {void} */
function openMenu(event) {
  const button = event.currentTarget;
  if (!panelMenu || !(button instanceof HTMLElement)) return;
  const { entries, run } = panelMenu;
  const rect = button.getBoundingClientRect();
  openContextMenu({
    x: rect.right,
    y: rect.bottom,
    triggerEl: button,
    anchorEl: button,
    ariaLabel: MENU_LABEL,
    items: menuRows(entries(), run),
  });
}

/**
 * Build the standalone ⋯ at the far right of the toolbar.
 * @param {HTMLElement} end - the toolbar's end group
 * @returns {void}
 */
function buildMenuButton(end) {
  const button = document.createElement('button');
  button.type = 'button';
  button.id = TOOLBAR_IDS.menu;
  button.className = 'osprey-icon-btn';
  button.setAttribute('aria-label', MENU_LABEL);
  button.setAttribute('aria-haspopup', 'menu');
  button.setAttribute('aria-expanded', 'false');
  button.append(icon('more'));
  button.addEventListener('click', openMenu);
  end.append(button);
}

/**
 * Wire the toolbar: the icons, the view switch, the standalone ⋯, and the
 * view and pipeline changes every control follows. Call once, at boot.
 * @returns {void}
 */
export function initToolbar() {
  const toggle = byId(TOOLBAR_IDS.toggle);
  if (toggle && !toggle.firstElementChild) toggle.append(icon('panel-left'));
  const field = byId(TOOLBAR_IDS.field);
  if (field && !field.querySelector('.osprey-ic')) field.prepend(icon('search'));
  const end = byId(TOOLBAR_IDS.end);
  if (end && !isEmbedded() && !byId(TOOLBAR_IDS.menu)) buildMenuButton(end);

  byId(TOOLBAR_IDS.views)?.addEventListener('click', (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const view = target?.closest('button[data-view]')?.getAttribute('data-view');
    if (view) location.hash = view;
  });
  state.on('viewChanged', refresh);
  state.on('pipelineChanged', refresh);
  refresh();
}

/**
 * Hand the search field and the ⋯ to the mounted view, emptied, until the
 * returned release gives them back (emptied again, with the result count).
 * @param {PanelMenu} menu - what the ⋯ lists and runs
 * @returns {() => void} release
 */
export function claimToolbar(menu) {
  const input = searchInput();
  if (input) input.value = '';
  searchClaimed = true;
  panelMenu = menu;
  refresh();
  return () => {
    const field = searchInput();
    if (field) field.value = '';
    searchClaimed = false;
    panelMenu = null;
    setResultCount('');
    refresh();
  };
}

/**
 * Show the result count as the toolbar's status text, or hide it when empty.
 * @param {string} html - trusted markup (graph-finder-render.js resultCountHtml)
 * @returns {void}
 */
export function setResultCount(html) {
  const count = byId(TOOLBAR_IDS.count);
  if (!count) return;
  count.innerHTML = html;
  count.hidden = html === '';
}

/**
 * Drive the toolbar's toggle from a drawn sidebar (initPanelSidebar), open on
 * a wide panel. Destroying it gives the toggle back, disabled.
 * @param {HTMLElement} sidebar - the `.osprey-panel-sidebar`
 * @param {HTMLElement|null} handle - its `.osprey-splitter`
 * @param {string} storageKey - the bare key; panel-sidebar.js scopes it
 * @returns {PanelSidebarApi}
 */
export function attachSidebar(sidebar, handle, storageKey) {
  const toggle = /** @type {HTMLButtonElement|null} */ (byId(TOOLBAR_IDS.toggle));
  if (toggle) toggle.disabled = false;
  sidebarAttached = true;
  const api = initPanelSidebar({ sidebar, handle, toggle, storageKey, defaultOpen: true });
  return {
    ...api,
    destroy() {
      api.destroy();
      sidebarAttached = false;
      byId(TOOLBAR_IDS.root)?.removeAttribute('data-open');
      refresh();
    },
  };
}
