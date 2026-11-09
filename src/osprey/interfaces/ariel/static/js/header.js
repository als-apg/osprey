// @ts-check
/**
 * ARIEL tile-bar header: the panel contributes `[search, compose button, ⋯ menu]`
 * to the web-terminal hub's tile bar (header-contrib.js).
 *
 * Store keys:
 * - reads `query` and `rerank`: a `query` change carrying `rerank:'when-stable'`
 *   is this module's own tile-bar report and leaves the search item alone; any
 *   other `query` change (saved search, hash navigation, alias) re-keys the
 *   search item so the hub builds a fresh field seeded with the new value.
 * - reads `railCollapsed`: the ⋯ menu's rail entry label.
 * - the ⋯ menu's Badges entry is checked by the badge-strip preference
 *   (display-prefs.js), toggles it, and is re-sent on its change.
 * - reads `filter`: a change made elsewhere that drops a value this module
 *   lifted re-keys the search item (the field is reseeded without its tokens).
 * - writes `{query, rerank:'when-stable'}` (search report), with `filter` (and
 *   `mailbox:'all'` when an author is lifted under Mine) in the same patch;
 *   `compose:{draft:null}` (compose button), `railCollapsed` (⋯ menu toggle).
 *
 * Search operators (query-ops.js): every report sends the query with its
 * `dim:value` tokens stripped. Complete tokens are lifted on the report; a
 * trailing token only once the reports settle (`RERANK_SETTLE_MS` after the
 * last one). The lifted values are this module's contribution to `filter`,
 * recomputed from the full text on every report and replacing the previous
 * one, so values set through the facet UI are kept.
 *
 * The whole contribution is re-sent on every relevant change; the hub renders
 * only the latest one. On invalid configuration the search item is omitted
 * (the list pane shows the config banner instead).
 */

import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';
import { getState, setState, subscribe } from './store.js';
import { promptSaveSearch } from './rail.js';
import { RERANK_SETTLE_MS } from './search.js';
import { holdsLifted, liftOperators, liftedPatch } from './query-ops.js';
import { getStripPref, onStripPrefChange, setStripPref } from './display-prefs.js';

/** Prefix of the search item's id; the suffix changes on every re-key. */
export const SEARCH_ID_PREFIX = 'ariel-search';
/** Compose button id. */
export const COMPOSE_ID = 'ariel-compose';
/** ⋯ menu id. */
export const MENU_ID = 'ariel-more';
/** ⋯ menu entry ids. */
export const MENU_TOGGLE_RAIL = 'toggle-rail';
export const MENU_SAVE_SEARCH = 'save-search';
export const MENU_TOGGLE_STRIP = 'toggle-strip';

/**
 * @typedef {Object} HeaderMount
 * @property {boolean} searchEnabled
 * @property {number} searchKey
 * @property {() => void} unsubscribe
 * @property {Array<{name: string}>|null} dimensions - capabilities.filter_dimensions
 * @property {import('./filter-state.js').Filter} lifted - This module's contribution to `filter`
 * @property {string} text - The latest reported search text
 * @property {ReturnType<typeof setTimeout>|null} settle - Lifts the trailing token
 * @property {boolean} writing - Set while this module writes the store
 */

/** @type {HeaderMount|null} */
let mounted = null;

/**
 * Build the whole contribution from the current state.
 * @returns {import('/design-system/js/header-contrib.js').HeaderItem[]}
 */
export function buildItems() {
  const state = getState();
  /** @type {import('/design-system/js/header-contrib.js').HeaderItem[]} */
  const items = [];
  if (mounted?.searchEnabled) {
    items.push({
      kind: 'search',
      id: `${SEARCH_ID_PREFIX}-${mounted.searchKey}`,
      placeholder: 'Search the logbook',
      value: state.query,
      priority: 2,
    });
  }
  items.push({ kind: 'button', id: COMPOSE_ID, label: 'Compose', tone: 'accent', priority: 3 });
  items.push({
    kind: 'menu',
    id: MENU_ID,
    label: 'Logbook options',
    priority: 1,
    items: [
      { id: MENU_TOGGLE_RAIL, label: state.railCollapsed ? 'Show sidebar' : 'Hide sidebar' },
      { id: MENU_SAVE_SEARCH, label: 'Save search…' },
      { id: MENU_TOGGLE_STRIP, label: 'Badges', checked: getStripPref() },
    ],
  });
  return items;
}

/** Send the whole contribution. */
function send() {
  contributeHeader(buildItems());
}

/**
 * Apply the tile bar's text: the stripped query plus the lifted values,
 * replacing the previous contribution.
 * @param {HeaderMount} mount
 * @param {boolean} final - Lift a trailing token too (the settle point)
 */
function applyText(mount, final) {
  const lift = liftOperators(mount.text, mount.dimensions, { final });
  const patch = liftedPatch(getState(), lift.filters, mount.lifted);
  mount.lifted = lift.filters;
  mount.writing = true;
  try {
    setState({ query: lift.query, rerank: 'when-stable', ...patch });
  } finally {
    mount.writing = false;
  }
}

/** @param {HeaderMount} mount */
function clearSettle(mount) {
  if (mount.settle !== null) clearTimeout(mount.settle);
  mount.settle = null;
}

/**
 * One tile-bar search report: apply it now, and again at the settle point.
 * @param {HeaderMount} mount
 * @param {string} text
 */
function report(mount, text) {
  clearSettle(mount);
  mount.text = text;
  applyText(mount, false);
  mount.settle = setTimeout(() => {
    mount.settle = null;
    if (mounted === mount) applyText(mount, true);
  }, RERANK_SETTLE_MS);
}

/**
 * Re-key the search item: the hub builds a fresh field seeded with `query`,
 * which holds no lifted token, so the contribution is forgotten.
 * @param {HeaderMount} mount
 */
function rekey(mount) {
  clearSettle(mount);
  mount.searchKey += 1;
  mount.lifted = {};
  mount.text = '';
}

/**
 * React to one tile-bar action.
 * @param {string} id - The contributed item's id.
 * @param {string|undefined} value - Search text, or the menu entry's id.
 */
export function handleHeaderAction(id, value) {
  if (!mounted) return;
  if (id.startsWith(`${SEARCH_ID_PREFIX}-`)) {
    if (!mounted.searchEnabled) return;
    report(mounted, value ?? '');
    return;
  }
  if (id === COMPOSE_ID) {
    setState({ compose: { draft: null } });
    return;
  }
  if (id !== MENU_ID) return;
  if (value === MENU_TOGGLE_RAIL) setState({ railCollapsed: !getState().railCollapsed });
  else if (value === MENU_SAVE_SEARCH) promptSaveSearch();
  else if (value === MENU_TOGGLE_STRIP) setStripPref(!getStripPref());
}

/**
 * Start contributing the tile-bar header.
 * @param {{capabilities?: {status?: string, filter_dimensions?: Array<{name: string}>|null}|null}} [options]
 * @returns {{destroy(): void}}
 */
export function initHeader({ capabilities = null } = {}) {
  if (mounted) {
    mounted.unsubscribe();
    clearSettle(mounted);
  }
  mounted = null;

  /** @type {HeaderMount} */
  const mount = {
    searchEnabled: capabilities?.status !== 'configuration_invalid',
    searchKey: 0,
    unsubscribe: () => {},
    dimensions: capabilities?.filter_dimensions ?? null,
    lifted: {},
    text: '',
    settle: null,
    writing: false,
  };
  mounted = mount;

  const offStore = subscribe((next, _prev, changed) => {
    let resend = false;
    if (changed.has('query')) {
      if (next.rerank !== 'when-stable') rekey(mount);
      resend = true;
    }
    if (changed.has('filter') && !mount.writing && !holdsLifted(next.filter, mount.lifted)) {
      rekey(mount);
      resend = true;
    }
    if (changed.has('railCollapsed')) resend = true;
    if (resend) send();
  });
  const offStrip = onStripPrefChange(() => send());
  mount.unsubscribe = () => {
    offStore();
    offStrip();
  };
  // header-contrib offers no unsubscribe; a replaced or destroyed mount ignores actions.
  onHeaderAction((id, value) => {
    if (mounted === mount) handleHeaderAction(id, value);
  });
  send();

  return {
    destroy() {
      mount.unsubscribe();
      clearSettle(mount);
      if (mounted === mount) mounted = null;
    },
  };
}
