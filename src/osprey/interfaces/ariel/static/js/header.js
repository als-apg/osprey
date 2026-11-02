// @ts-check
/**
 * ARIEL embedded tile-bar header: when the panel runs inside the web-terminal
 * hub, its own `.header` is hidden and the panel contributes
 * `[search, compose button, ⋯ menu]` to the hub's tile bar instead
 * (header-contrib.js). A no-op standalone.
 *
 * Store keys:
 * - reads `query` and `rerank`: a `query` change carrying `rerank:'when-stable'`
 *   is this module's own tile-bar report and leaves the search item alone; any
 *   other `query` change (saved search, hash navigation, alias) re-keys the
 *   search item so the hub builds a fresh field seeded with the new value.
 * - reads `railCollapsed`: the ⋯ menu's rail entry label.
 * - writes `{query, rerank:'when-stable'}` (search report), `compose:{draft:null}`
 *   (compose button), `railCollapsed` (⋯ menu toggle).
 *
 * The whole contribution is re-sent on every relevant change; the hub renders
 * only the latest one. On invalid configuration the search item is omitted
 * (the list pane shows the config banner instead).
 */

import { isEmbedded } from '/design-system/js/frame-params.js';
import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';
import { getState, setState, subscribe } from './store.js';
import { promptSaveSearch } from './rail.js';

/** Prefix of the search item's id; the suffix changes on every re-key. */
export const SEARCH_ID_PREFIX = 'ariel-search';
/** Compose button id. */
export const COMPOSE_ID = 'ariel-compose';
/** ⋯ menu id. */
export const MENU_ID = 'ariel-more';
/** ⋯ menu entry ids. */
export const MENU_TOGGLE_RAIL = 'toggle-rail';
export const MENU_SAVE_SEARCH = 'save-search';

/**
 * @typedef {Object} HeaderMount
 * @property {boolean} searchEnabled
 * @property {number} searchKey
 * @property {() => void} unsubscribe
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
    ],
  });
  return items;
}

/** Send the whole contribution. */
function send() {
  contributeHeader(buildItems());
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
    setState({ query: value ?? '', rerank: 'when-stable' });
    return;
  }
  if (id === COMPOSE_ID) {
    setState({ compose: { draft: null } });
    return;
  }
  if (id !== MENU_ID) return;
  if (value === MENU_TOGGLE_RAIL) setState({ railCollapsed: !getState().railCollapsed });
  else if (value === MENU_SAVE_SEARCH) promptSaveSearch();
}

/**
 * Start contributing the tile-bar header. No-op standalone.
 * @param {{capabilities?: {status?: string}|null}} [options]
 * @returns {{destroy(): void}}
 */
export function initHeader({ capabilities = null } = {}) {
  if (mounted) mounted.unsubscribe();
  mounted = null;
  if (!isEmbedded()) return { destroy() {} };

  /** @type {HeaderMount} */
  const mount = {
    searchEnabled: capabilities?.status !== 'configuration_invalid',
    searchKey: 0,
    unsubscribe: () => {},
  };
  mounted = mount;

  mount.unsubscribe = subscribe((next, _prev, changed) => {
    let resend = false;
    if (changed.has('query')) {
      if (next.rerank !== 'when-stable') mount.searchKey += 1;
      resend = true;
    }
    if (changed.has('railCollapsed')) resend = true;
    if (resend) send();
  });
  // header-contrib offers no unsubscribe; a replaced or destroyed mount ignores actions.
  onHeaderAction((id, value) => {
    if (mounted === mount) handleHeaderAction(id, value);
  });
  send();

  return {
    destroy() {
      mount.unsubscribe();
      if (mounted === mount) mounted = null;
    },
  };
}
