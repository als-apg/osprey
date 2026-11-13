// @ts-check
/**
 * OSPREY Channel Finder — the panel toolbar.
 *
 * The one 40 px row at the top of the panel (DESIGN.md §12.3; the markup is in
 * index.html):
 *
 *   [sidebar toggle] [Explore | Feedback] …… [result count] [search]
 *
 * Beside a PV directory the switch reads [Curated | <directory> | Feedback]
 * (app.js writes it); a PV or IOC page presses the directory's button.
 *
 * The view switch belongs to the shell: a press sets the hash and app.js
 * routes on it; the pressed button follows `state.activeView`. The rest
 * belongs to a view with a search and filters. While it is mounted it claims
 * the search field ({@link claimToolbar}) and writes the result count
 * ({@link setResultCount}); while its filters are drawn they drive the sidebar
 * toggle ({@link attachSidebar}), and while they are not, a view may hand the
 * toolbar its own reason for the disabled toggle ({@link setFilterReason}).
 * On a page that is part of All PVs and claims
 * nothing, the search is the toolbar's own: Enter opens All PVs with the words.
 * In every other view those controls keep their slots, disabled, with a
 * tooltip that says why, so a change of view moves nothing.
 *
 * The panel's options are not here: they are the tile ⋯'s "This panel" section
 * (graph-finder-verbs.js publishMenu), and a tile has one ⋯.
 */

import { icon } from '/design-system/js/icons.js';
import { initPanelSidebar } from '/design-system/js/panel-sidebar.js';
import { writeTitle } from '/design-system/js/tooltip.js';
import { allPvsHash, familyOf, formatRoute } from './route.js';
import { state } from './state.js';

/** @typedef {import('/design-system/js/panel-sidebar.js').PanelSidebarApi} PanelSidebarApi */

/** The toolbar's elements, by id (index.html). */
export const TOOLBAR_IDS = Object.freeze({
  root: 'cf-panel',
  toggle: 'cf-sidebar-toggle',
  views: 'cf-views',
  count: 'cf-result-count',
  field: 'cf-search-field',
  search: 'cf-search',
});

const SHOW_SIDEBAR = 'Show sidebar';

/** Why a control is disabled: under Feedback (when the curated tab has it), or anywhere else. */
const REASONS = Object.freeze({
  filters: ['Filters apply to', 'No filters for this view'],
  search: ['Search applies to', 'No search for this view'],
});

/** Whether a view holds the search field. */
let searchClaimed = false;

/** Whether the search field is the toolbar's own: unclaimed, on a page of All PVs. */
function searchRedirects() {
  return !searchClaimed && familyOf(state.activeView) === 'allpvs';
}
/** Whether a drawn sidebar drives the toggle. */
let sidebarAttached = false;
/**
 * The mounted view's own reason for the disabled filter toggle, or null for
 * the toolbar's; dropped on every view change.
 * @type {string|null}
 */
let filterReason = null;

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
 * controls point back to the curated tab (Explore, or Curated beside a
 * directory), but only where it has them: the graph paradigm is the one with
 * filters and a search.
 * @param {keyof typeof REASONS} kind
 * @returns {string}
 */
function reason(kind) {
  const [feedback, other] = REASONS[kind];
  if (state.activeView !== 'feedback' || state.pipelineType !== 'graph') return other;
  return `${feedback} ${state.directory ? 'Curated' : 'Explore'}`;
}

/** Bring every control in line with the view and with what is claimed. @returns {void} */
function refresh() {
  const pressed = familyOf(state.activeView);
  for (const btn of byId(TOOLBAR_IDS.views)?.querySelectorAll('button[data-view]') ?? []) {
    btn.setAttribute('aria-pressed', String(btn.getAttribute('data-view') === pressed));
  }

  const enabled = searchClaimed || searchRedirects();
  const input = searchInput();
  if (input) input.disabled = !enabled;
  const field = byId(TOOLBAR_IDS.field);
  if (field) writeTitle(field, enabled ? '' : reason('search'));

  // A drawn sidebar keeps the toggle in step itself (panel-sidebar.js).
  const toggle = /** @type {HTMLButtonElement|null} */ (byId(TOOLBAR_IDS.toggle));
  if (toggle && !sidebarAttached) {
    toggle.disabled = true;
    toggle.setAttribute('aria-label', SHOW_SIDEBAR);
    toggle.removeAttribute('aria-expanded');
    toggle.removeAttribute('aria-controls');
    writeTitle(toggle, filterReason ?? reason('filters'));
  }
}

/**
 * Wire the toolbar: the icons, the view switch, the search's own Enter, and
 * the view and pipeline changes every control follows. Call once, at boot.
 * @returns {void}
 */
export function initToolbar() {
  const toggle = byId(TOOLBAR_IDS.toggle);
  if (toggle && !toggle.firstElementChild) toggle.append(icon('panel-left'));
  const field = byId(TOOLBAR_IDS.field);
  if (field && !field.querySelector('.osprey-ic')) field.prepend(icon('search'));

  byId(TOOLBAR_IDS.views)?.addEventListener('click', (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const view = target?.closest('button[data-view]')?.getAttribute('data-view');
    // The active view's own button keeps the route that view wrote.
    if (view && view !== state.activeView) location.hash = formatRoute({ view, params: {} });
  });
  searchInput()?.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' || !searchRedirects()) return;
    location.hash = allPvsHash(/** @type {HTMLInputElement} */ (event.target).value);
  });
  state.on('viewChanged', () => {
    filterReason = null;
    refresh();
  });
  state.on('pipelineChanged', refresh);
  refresh();
}

/**
 * Give the disabled filter toggle the mounted view's own reason, kept until
 * the view changes; null gives it back the toolbar's. The toolbar is the one
 * writer of that tooltip.
 * @param {string|null} text
 * @returns {void}
 */
export function setFilterReason(text) {
  filterReason = text || null;
  refresh();
}

/**
 * Hand the search field to the mounted view, emptied, until the returned
 * release gives it back (emptied again, with the result count).
 * @returns {() => void} release
 */
export function claimToolbar() {
  const input = searchInput();
  if (input) input.value = '';
  searchClaimed = true;
  refresh();
  return () => {
    const field = searchInput();
    if (field) field.value = '';
    searchClaimed = false;
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
