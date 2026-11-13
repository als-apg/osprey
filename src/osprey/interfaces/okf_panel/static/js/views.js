// @ts-check
/*
 * OKF Knowledge Panel — the panel's two views and the switch between them.
 *
 *   knowledge   the contents list and the reading pane (the default)
 *   graph       the facility graph (graph-view.js)
 *
 * The Graph view exists only when the panel serves a graph, which the page
 * states as `#app[data-graph="true"]`; without one the switcher stays hidden
 * and the panel is the Knowledge view alone. The toolbar holds a sidebar
 * toggle, a context and a chip per view; this module shows the open view's.
 */

/** @typedef {'knowledge' | 'graph'} PanelView */

/** What the search field says in each view. */
const SEARCH_LABEL = { knowledge: 'Search concepts', graph: 'Search nodes' };

/** Element ids shown only in the one view. */
const ONLY_IN = {
  knowledge: ['knowledge-view', 'contents-toggle', 'crumb', 'bundle-health'],
  graph: ['graph-view', 'graph-toggle', 'graph-context', 'graph-chip'],
};

/** @type {PanelView} */
let current = 'knowledge';

/**
 * Whether the panel serves a facility graph.
 * @returns {boolean}
 */
export function graphAvailable() {
  const app = document.getElementById('app');
  return !!app && app.dataset.graph === 'true';
}

/**
 * The view the panel shows.
 * @returns {PanelView}
 */
export function currentView() {
  return current;
}

/**
 * Show one view and its toolbar parts. Asking for the Graph view of a panel
 * that serves no graph changes nothing.
 * @param {PanelView} view
 * @returns {boolean} whether the view changed
 */
export function showView(view) {
  if (view === current || (view === 'graph' && !graphAvailable())) return false;
  current = view;
  for (const name of /** @type {PanelView[]} */ (['knowledge', 'graph'])) {
    for (const id of ONLY_IN[name]) {
      const node = document.getElementById(id);
      if (node) node.hidden = name !== view;
    }
  }
  for (const button of document.querySelectorAll('#okf-views button')) {
    button.setAttribute('aria-pressed', String(button.getAttribute('data-view') === view));
  }
  const search = /** @type {HTMLInputElement | null} */ (document.getElementById('search-input'));
  if (search) {
    search.placeholder = SEARCH_LABEL[view] + '…';
    search.setAttribute('aria-label', SEARCH_LABEL[view]);
  }
  return true;
}

/**
 * Wire the view switcher. It is shown only when the panel serves a graph.
 * Call once at boot; it starts on the Knowledge view.
 * @param {{onSwitch: (view: PanelView) => void}} handles `onSwitch`: the
 *   switcher asked for the other view (nothing has changed yet)
 */
export function initViews({ onSwitch }) {
  current = 'knowledge';
  const switcher = document.getElementById('okf-views');
  if (!switcher || !graphAvailable()) return;
  switcher.hidden = false;
  switcher.addEventListener('click', (ev) => {
    const button = ev.target instanceof Element ? ev.target.closest('button') : null;
    const view = button && button.getAttribute('data-view');
    if ((view === 'knowledge' || view === 'graph') && view !== current) onSwitch(view);
  });
}
