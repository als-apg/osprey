// @ts-check
/**
 * ARIEL list paging
 *
 * Owns the paging state of the one list run in progress; no DOM. Browse
 * (`/api/entries`, newest first) continues with the previous response's
 * `next_cursor` and ends when there is none. Search (`/api/search`) pages by
 * offset in PAGE_SIZE steps and drops ids already shown, because a reranked
 * first page and the fast-ranked later pages may overlap; it ends when a
 * response says `has_more` is not true.
 *
 * Every start and `reset()` aborts the previous run's request and forgets its
 * state, so a superseded response resolves to null and is never painted.
 * While a search's phase-2 rerank is pending, `loadMore()` does nothing: the
 * reranked page 1 replaces the fast one and restarts the offset.
 */

import { entriesApi } from './api.js';
import { mergeFilter, requestFilter, splitFilters } from './filter-state.js';
import { isAbort, runSearch, searchPage } from './search.js';

/** @typedef {import('./filter-state.js').Filter} Filter */
/** @typedef {import('./search.js').SearchRequest} SearchRequest */
/** @typedef {import('./search.js').SearchHooks} SearchHooks */
/** @typedef {import('./advanced-options.js').Capabilities} Capabilities */

/** Rows per page, for browse and search alike. */
export const PAGE_SIZE = 20;

/**
 * @typedef {Object} BrowsePaging
 * @property {'browse'} kind
 * @property {Filter} filter
 * @property {string|null} cursor - The next page's cursor
 * @property {boolean} done - No next page
 * @property {number|null} total - The filtered count
 */

/**
 * @typedef {Object} SearchPaging
 * @property {'search'} kind
 * @property {SearchRequest} request
 * @property {Capabilities|null} capabilities
 * @property {number} offset - Rows the server has served so far
 * @property {number} total
 * @property {boolean} totalExact
 * @property {boolean} hasMore
 * @property {Set<string>} shownIds
 * @property {boolean} held - Phase-2 rerank pending
 */

/**
 * @typedef {Object} PageInfo
 * @property {'browse'|'search'|null} kind
 * @property {number|null} total
 * @property {boolean} totalExact
 * @property {boolean} held - A rerank is pending: "load more" waits
 * @property {boolean} loading - A page is in flight
 */

/** @type {BrowsePaging|SearchPaging|null} */
let paging = null;

/** @type {AbortController|null} */
let controller = null;

let loading = false;

/**
 * Abort the previous run and open a new one.
 * @param {BrowsePaging|SearchPaging} next
 * @returns {AbortSignal}
 */
function begin(next) {
  controller?.abort();
  controller = new AbortController();
  paging = next;
  loading = true;
  return controller.signal;
}

/**
 * The filter the browse listing applies: dimension values still held among
 * the Filters & Options knobs under the request filter (mailbox scope
 * included), the same merge the search driver makes.
 * @param {import('./store.js').State} state
 * @returns {Filter}
 */
export function listingFilter(state) {
  return mergeFilter(splitFilters(state.filters).filter, requestFilter(state) ?? {});
}

/**
 * A total as the header shows it: exact as is, an inexact one with more
 * pages to come as a lower bound (`40+`), any other inexact one as an
 * estimate (`≈40`).
 * @param {number|null} total
 * @param {boolean} exact
 * @param {boolean} more
 * @returns {string}
 */
export function formatTotal(total, exact, more) {
  const n = String(total ?? 0);
  if (exact) return n;
  return more ? `${n}+` : `\u2248${n}`;
}

/** @returns {string} The current run's total as the header shows it. */
export function totalText() {
  const info = pageInfo();
  return formatTotal(info.total, info.totalExact, canLoadMore());
}

/** Abort any request in flight and forget the paging state. */
export function reset() {
  controller?.abort();
  controller = null;
  paging = null;
  loading = false;
}

/**
 * Take a search response's total into the state. An inexact total is a lower
 * bound or an estimate, so a later page never lowers it.
 * @param {SearchPaging} state
 * @param {any} results
 */
function takeTotal(state, results) {
  const total = Number(results?.total ?? results?.total_results ?? 0);
  state.totalExact = results?.total_exact !== false;
  state.total = state.totalExact ? total : Math.max(state.total, total);
}

/**
 * Fetch the browse page after `mine.cursor` and advance the cursor.
 * @param {BrowsePaging} mine
 * @param {AbortSignal|undefined} signal
 * @returns {Promise<any>} The response; null when superseded
 */
async function browsePage(mine, signal) {
  const result = /** @type {any} */ (
    await entriesApi.list({
      cursor: mine.cursor ?? undefined,
      pageSize: PAGE_SIZE,
      filters: mine.filter,
      view: 'row',
      signal,
    })
  );
  if (paging !== mine) return null;
  mine.cursor = result?.next_cursor ?? null;
  mine.done = !mine.cursor;
  return result;
}

/**
 * Fetch the first browse page.
 * @param {Filter} filter - The C-FILTER the listing applies
 * @returns {Promise<{entries: any[], total: number|null}|null>} null when superseded
 */
export async function startListing(filter) {
  /** @type {BrowsePaging} */
  const mine = { kind: 'browse', filter, cursor: null, done: false, total: null };
  const signal = begin(mine);
  try {
    const result = await browsePage(mine, signal);
    if (result === null) return null;
    mine.total = typeof result?.total === 'number' ? result.total : null;
    return { entries: result?.entries ?? [], total: mine.total };
  } catch (error) {
    if (paging !== mine || isAbort(error)) return null;
    mine.done = true;
    throw error;
  } finally {
    if (paging === mine) loading = false;
  }
}

/**
 * Run page 1 of a search through the driver (two-phase rerank included). The
 * hooks are called as runSearch calls them, after the paging state took the
 * response, and never for a superseded run.
 * @param {SearchRequest} request - Without offset, page size or view
 * @param {SearchHooks} hooks
 * @returns {Promise<void>}
 */
export async function startSearch(request, hooks) {
  /** @type {SearchPaging} */
  const mine = {
    kind: 'search',
    request,
    capabilities: hooks.capabilities,
    offset: 0,
    total: 0,
    totalExact: true,
    hasMore: false,
    shownIds: new Set(),
    held: false,
  };
  const signal = begin(mine);
  try {
    await runSearch(
      { ...request, offset: 0, pageSize: PAGE_SIZE, view: 'row', signal },
      {
        ...hooks,
        onStatus: (status) => {
          if (paging !== mine) return;
          mine.held = status === 'reranking';
          hooks.onStatus(status);
        },
        onResults: (results) => {
          if (paging !== mine) return;
          loading = false;
          const r = /** @type {any} */ (results);
          const entries = r?.entries ?? [];
          mine.offset = entries.length;
          mine.shownIds = new Set(entries.map((/** @type {any} */ e) => e.entry_id));
          mine.hasMore = r?.has_more === true;
          mine.total = 0;
          takeTotal(mine, r);
          hooks.onResults(results);
        },
      }
    );
  } catch (error) {
    if (paging !== mine || isAbort(error)) return;
    throw error;
  } finally {
    if (paging === mine) loading = false;
  }
}

/**
 * Whether the run has another page: browse has a `next_cursor`, search's last
 * response said `has_more`.
 * @returns {boolean}
 */
export function canLoadMore() {
  if (!paging) return false;
  return paging.kind === 'browse' ? !paging.done : paging.hasMore;
}

/** @returns {PageInfo} */
export function pageInfo() {
  if (!paging) return { kind: null, total: null, totalExact: true, held: false, loading };
  return {
    kind: paging.kind,
    total: paging.total,
    totalExact: paging.kind === 'search' ? paging.totalExact : true,
    held: paging.kind === 'search' && paging.held,
    loading,
  };
}

/**
 * Fetch the next page. Nothing is requested at the end, while a page is in
 * flight, or while a rerank is pending.
 * @returns {Promise<{entries: any[]}|null>} The new rows (search: only ids not
 *   shown yet); null when nothing ran or the run was superseded
 */
export async function loadMore() {
  const mine = paging;
  if (!mine || !canLoadMore() || loading) return null;
  if (mine.kind === 'search' && mine.held) return null;
  const signal = controller?.signal;
  loading = true;
  try {
    if (mine.kind === 'browse') {
      const result = await browsePage(mine, signal);
      return result === null ? null : { entries: result?.entries ?? [] };
    }
    const result = await searchPage(
      { ...mine.request, offset: mine.offset, pageSize: PAGE_SIZE, view: 'row', signal },
      { capabilities: mine.capabilities }
    );
    if (paging !== mine) return null;
    const entries = /** @type {any[]} */ (result?.entries ?? []);
    mine.offset += entries.length;
    const fresh = entries.filter((e) => !mine.shownIds.has(e.entry_id));
    for (const e of fresh) mine.shownIds.add(e.entry_id);
    mine.hasMore = result?.has_more === true && entries.length > 0;
    takeTotal(mine, result);
    return { entries: fresh };
  } catch (error) {
    if (paging !== mine || isAbort(error)) return null;
    throw error;
  } finally {
    if (paging === mine) loading = false;
  }
}
