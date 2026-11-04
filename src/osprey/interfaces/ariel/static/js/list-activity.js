// @ts-check
/**
 * ARIEL list activity: the agent's writes reach the list without moving it.
 *
 * The host relays the agent's ARIEL tool calls as same-origin
 * `osprey-agent-activity {tool, detail, ts}` messages (to the embedded
 * iframe, or to the pop-out window). `entry_create` and `entry_publish` both
 * name a new row in `detail`: publish writes a new facility entry.
 *
 * Each frame asks one question — is that row on page 1 of the current view? —
 * by a side fetch through api.js, never reranked, built with list.js's own
 * request builders. On a hit:
 *   - browsing, the list at the top and only page 1 loaded: `refresh()` the
 *     list, then flash the row;
 *   - otherwise (scrolled, paged, or a search view, whose ranked order a
 *     refresh would reshuffle): a create adds to the "n new" pill, whose click
 *     scrolls to the top and refreshes; a publish never counts as new.
 *
 * Frames coalesce into a set of pending ids: one fetch in flight plus one
 * trailing for the ids that arrived meanwhile, and every pending id counts. A
 * view change (query, mode, filters, mailbox, Mine name) drops pending ids and
 * any answer still in flight, and clears the pill. Drafts, and Mine without a
 * name, ignore frames. Nothing here writes `sel`, splices rows into the DOM or
 * restarts paging other than through `refresh()`.
 */

import { entriesApi } from './api.js';
import { searchPage } from './search.js';
import { PAGE_SIZE, listingFilter } from './paging.js';
import { refresh, searchRequest } from './list.js';
import { getState, subscribe } from './store.js';
import { flashElement } from '/design-system/js/highlight.js';

/** @typedef {import('./store.js').State} State */
/** @typedef {import('./advanced-options.js').Capabilities} Capabilities */

/** The host's relay message type. */
export const ACTIVITY_MESSAGE = 'osprey-agent-activity';

/** The tools whose `detail` is the id of a new row. */
const ROW_TOOLS = new Set(['entry_create', 'entry_publish']);

/** Store keys that change what the list shows. */
const VIEW_KEYS = ['query', 'mode', 'filters', 'filter', 'mailbox', 'mineAuthor'];

/** @type {Capabilities|null} */
let capabilities = null;

/** @type {Map<string, string>} Pending row ids, each with the tool that named it. */
let pending = new Map();

/** A side fetch (and what follows it) is running. */
let busy = false;

/** Bumped on every view change; an answer for an older view is dropped. */
let viewSeq = 0;

/** Rows the agent created that the list does not show yet. */
let newCount = 0;

/** @type {HTMLButtonElement|null} */
let pill = null;

/** @type {(() => void)|null} */
let stop = null;

/** @returns {HTMLElement|null} The list body (`#ariel-list-body`). */
function listBody() {
  return document.getElementById('ariel-list-body');
}

/**
 * Whether the state is a view the agent's rows can land in.
 * @param {State} state
 * @returns {boolean}
 */
function watching(state) {
  if (state.mailbox === 'drafts') return false;
  return !(state.mailbox === 'mine' && !state.mineAuthor);
}

/**
 * Whether the list runs a search for `state` (as list.js decides: a query,
 * and search not blocked by an invalid configuration).
 * @param {State} state
 * @returns {boolean}
 */
function isSearchView(state) {
  const blocked = /** @type {any} */ (capabilities)?.status === 'configuration_invalid';
  return Boolean((state.query ?? '').trim()) && !blocked;
}

/**
 * The ids on page 1 of the view `state` shows, from one un-reranked request.
 * @param {State} state
 * @returns {Promise<Set<string>>}
 */
async function pageOneIds(state) {
  const response = isSearchView(state)
    ? await searchPage(
        { ...searchRequest(state, capabilities), offset: 0, pageSize: PAGE_SIZE, view: 'row' },
        { capabilities }
      )
    : await entriesApi.list({ pageSize: PAGE_SIZE, filters: listingFilter(state), view: 'row' });
  const entries = /** @type {any} */ (response)?.entries ?? [];
  return new Set(entries.map((/** @type {any} */ e) => String(e.entry_id)));
}

/** @param {number} n */
function setCount(n) {
  newCount = n;
  if (!pill) return;
  pill.textContent = `${n} new`;
  pill.hidden = n === 0;
}

/** Whether the list shows its top and nothing past page 1. */
function atTopOfPageOne() {
  const body = listBody();
  if (!body) return false;
  return body.scrollTop < 1 && !body.querySelector('.ariel-rows[data-paged]');
}

/** @param {string} id */
function flashRow(id) {
  const rows = /** @type {HTMLElement[]} */ (
    Array.from(listBody()?.querySelectorAll('[data-entry-id]') ?? [])
  );
  const row = rows.find((el) => el.dataset.entryId === id);
  if (row) flashElement(row);
}

/**
 * Act on the pending ids page 1 holds.
 * @param {Map<string, string>} batch
 * @param {Set<string>} ids
 * @param {State} state
 * @param {number} seq
 */
async function apply(batch, ids, state, seq) {
  const hits = [...batch].filter(([id]) => ids.has(id));
  if (hits.length === 0) return;
  if (!isSearchView(state) && atTopOfPageOne()) {
    await refresh();
    if (seq !== viewSeq) return;
    setCount(0);
    hits.forEach(([id]) => flashRow(id));
    return;
  }
  const created = hits.filter(([, tool]) => tool === 'entry_create').length;
  if (created > 0) setCount(newCount + created);
}

/** Run the side fetch for the pending ids, then once more for any that came meanwhile. */
async function pump() {
  if (busy || pending.size === 0) return;
  busy = true;
  const batch = pending;
  pending = new Map();
  const seq = viewSeq;
  const state = getState();
  try {
    const ids = await pageOneIds(state);
    if (seq === viewSeq) await apply(batch, ids, state, seq);
  } catch (error) {
    console.warn('Agent activity: page-1 check failed:', error);
  } finally {
    busy = false;
  }
  pump();
}

/** @param {MessageEvent} event */
function onMessage(event) {
  if (event.origin !== window.location.origin) return;
  const data = event.data;
  if (!data || data.type !== ACTIVITY_MESSAGE || !ROW_TOOLS.has(data.tool)) return;
  const id = data.detail == null ? '' : String(data.detail);
  if (!id || !watching(getState())) return;
  pending.set(id, data.tool);
  pump();
}

/**
 * The store subscriber: a view change starts over.
 * @param {State} _state
 * @param {State} _prev
 * @param {Set<string>} changed
 */
function onStoreChange(_state, _prev, changed) {
  if (!VIEW_KEYS.some((k) => changed.has(k))) return;
  viewSeq++;
  pending = new Map();
  setCount(0);
}

/** The "n new" pill, placed above the list body. */
function buildPill() {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'osprey-btn ariel-new-pill';
  button.dataset.newPill = '';
  button.setAttribute('aria-live', 'polite');
  button.hidden = true;
  button.addEventListener('click', () => {
    const body = listBody();
    if (body) body.scrollTop = 0;
    setCount(0);
    refresh();
  });
  listBody()?.before(button);
  return button;
}

/**
 * Start following agent activity. Call after `initList`, which builds the
 * list body the pill sits above. Calling again replaces the previous wiring.
 * @param {{capabilities?: Capabilities|null}} [options]
 * @returns {() => void} Stops following and removes the pill
 */
export function initListActivity({ capabilities: caps = null } = {}) {
  stop?.();
  capabilities = caps;
  pending = new Map();
  busy = false;
  viewSeq++;
  newCount = 0;
  pill = buildPill();
  const unsubscribe = subscribe(onStoreChange);
  window.addEventListener('message', onMessage);
  stop = () => {
    window.removeEventListener('message', onMessage);
    unsubscribe();
    pill?.remove();
    pill = null;
    pending = new Map();
    viewSeq++;
    stop = null;
  };
  return stop;
}
