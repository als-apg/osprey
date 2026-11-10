// @ts-check
/**
 * ARIEL list paging (paging.js):
 *   npx vitest run tests/interfaces/ariel/paging.test.mjs
 *
 * paging.js owns the paging state of one list run and no DOM. Browse continues
 * from `next_cursor` and ends without one; search pages by offset in PAGE_SIZE
 * steps, drops ids already shown, and ends when `has_more` is not true. A
 * start or `reset()` aborts the previous run, whose late response resolves to
 * null. While a phase-2 rerank is pending, `loadMore()` requests nothing.
 *
 * Runs under happy-dom. Only api.js is mocked (the network boundary); the
 * search driver runs for real. Module state is per import, so every case
 * resets modules.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    searchApi: { ...actual.searchApi, search: vi.fn() },
    entriesApi: { ...actual.entriesApi, list: vi.fn() },
  };
});

const API_PATH = '../../../src/osprey/interfaces/ariel/static/js/api.js';
const PAGING_PATH = '../../../src/osprey/interfaces/ariel/static/js/paging.js';

/** One keyword mode, no rerank. */
const KEYWORD = /** @type {any} */ ({
  categories: {
    direct: { label: 'Direct', modes: [{ name: 'keyword', label: 'Keyword', parameters: [] }] },
  },
  default_mode: 'keyword',
  shared_parameters: [],
});

/** One hybrid mode that reranks by default. */
const HYBRID = /** @type {any} */ ({
  categories: {
    direct: {
      label: 'Direct',
      modes: [
        {
          name: 'hybrid',
          label: 'Hybrid',
          parameters: [{ name: 'rerank', label: 'Rerank', type: 'bool', default: true }],
        },
      ],
    },
  },
  default_mode: 'hybrid',
  shared_parameters: [],
});

/** @type {any} */
let api;
/** @type {any} */
let paging;

/** @param {string[]} ids */
const rows = (ids) => ids.map((id) => ({ entry_id: id }));

/**
 * @param {string[]} ids
 * @param {string|null} next_cursor
 */
const listPage = (ids, next_cursor = null) => ({ entries: rows(ids), total: 9, next_cursor });

/**
 * @param {string[]} ids
 * @param {{total?: number, total_exact?: boolean, has_more?: boolean}} [opts]
 */
const searchPage = (ids, { total = 9, total_exact = true, has_more = false } = {}) => ({
  entries: rows(ids),
  total_results: total,
  total,
  total_exact,
  has_more,
});

/** Hooks that ignore every report. */
const quiet = (capabilities = KEYWORD) => ({
  capabilities,
  onStatus: () => {},
  onResults: () => {},
});

/** @returns {{promise: Promise<any>, resolve: (v: any) => void}} */
function deferred() {
  /** @type {(v: any) => void} */
  let resolve = () => {};
  const promise = new Promise((r) => (resolve = r));
  return { promise, resolve };
}

beforeEach(async () => {
  vi.resetModules();
  vi.clearAllMocks();
  api = await import(API_PATH);
  paging = await import(PAGING_PATH);
  vi.mocked(api.entriesApi.list).mockReset();
  vi.mocked(api.searchApi.search).mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('browse', () => {
  test('the first page asks for rows with the filter and no cursor', async () => {
    vi.mocked(api.entriesApi.list).mockResolvedValueOnce(listPage(['A', 'B'], 'C1'));
    const first = await paging.startListing({ author: ['bob'] });
    expect(first).toEqual({ entries: rows(['A', 'B']), total: 9 });
    const sent = vi.mocked(api.entriesApi.list).mock.calls[0][0];
    expect(sent).toMatchObject({
      pageSize: paging.PAGE_SIZE,
      filters: { author: ['bob'] },
      view: 'row',
    });
    expect(sent.cursor).toBeUndefined();
    expect(sent.signal).toBeInstanceOf(AbortSignal);
    expect(paging.canLoadMore()).toBe(true);
  });

  test('loadMore continues from next_cursor with the same filter', async () => {
    vi.mocked(api.entriesApi.list)
      .mockResolvedValueOnce(listPage(['A'], 'C1'))
      .mockResolvedValueOnce(listPage(['B'], 'C2'));
    await paging.startListing({ source_system: ['ops'] });
    const next = await paging.loadMore();
    expect(next).toEqual({ entries: rows(['B']) });
    expect(vi.mocked(api.entriesApi.list).mock.calls[1][0]).toMatchObject({
      cursor: 'C1',
      filters: { source_system: ['ops'] },
      view: 'row',
    });
    expect(paging.canLoadMore(), 'C2 is another page').toBe(true);
  });

  test('no cursor is the end: loadMore requests nothing', async () => {
    vi.mocked(api.entriesApi.list).mockResolvedValueOnce(listPage(['A'], null));
    await paging.startListing({});
    expect(paging.canLoadMore()).toBe(false);
    expect(await paging.loadMore()).toBeNull();
    expect(api.entriesApi.list).toHaveBeenCalledTimes(1);
  });

  test('a page in flight blocks a second loadMore', async () => {
    const later = deferred();
    vi.mocked(api.entriesApi.list)
      .mockResolvedValueOnce(listPage(['A'], 'C1'))
      .mockImplementationOnce(() => later.promise);
    await paging.startListing({});
    const first = paging.loadMore();
    expect(await paging.loadMore(), 'still loading').toBeNull();
    later.resolve(listPage(['B'], null));
    await first;
    expect(api.entriesApi.list).toHaveBeenCalledTimes(2);
  });
});

describe('search', () => {
  test('page 1 runs through the driver at offset 0, PAGE_SIZE rows, view row', async () => {
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(searchPage(['A'], { has_more: true }));
    await paging.startSearch({ query: 'q', mode: 'keyword', filters: {} }, quiet());
    expect(vi.mocked(api.searchApi.search).mock.calls[0][0]).toMatchObject({
      offset: 0,
      maxResults: paging.PAGE_SIZE,
      view: 'row',
    });
    expect(paging.canLoadMore()).toBe(true);
    expect(paging.pageInfo()).toMatchObject({ kind: 'search', total: 9, totalExact: true });
  });

  test('loadMore advances the offset by the served rows and drops ids already shown', async () => {
    vi.mocked(api.searchApi.search)
      .mockResolvedValueOnce(searchPage(['A', 'B'], { has_more: true }))
      .mockResolvedValueOnce(searchPage(['B', 'C'], { has_more: true }))
      .mockResolvedValueOnce(searchPage(['D'], { has_more: false }));
    await paging.startSearch({ query: 'q', mode: 'keyword', filters: {} }, quiet());

    expect(await paging.loadMore()).toEqual({ entries: rows(['C']) });
    expect(vi.mocked(api.searchApi.search).mock.calls[1][0]).toMatchObject({
      query: 'q',
      offset: 2,
      maxResults: paging.PAGE_SIZE,
    });
    expect(await paging.loadMore()).toEqual({ entries: rows(['D']) });
    expect(vi.mocked(api.searchApi.search).mock.calls[2][0].offset, 'served, not deduped').toBe(4);
    expect(paging.canLoadMore(), 'has_more false: the end').toBe(false);
    expect(await paging.loadMore()).toBeNull();
    expect(api.searchApi.search).toHaveBeenCalledTimes(3);
  });

  test('has_more missing is the end', async () => {
    vi.mocked(api.searchApi.search).mockResolvedValueOnce({
      entries: rows(['A']),
      total_results: 1,
    });
    await paging.startSearch({ query: 'q', mode: 'keyword', filters: {} }, quiet());
    expect(paging.canLoadMore()).toBe(false);
  });

  test('an inexact total never goes down; the formatted total marks it', async () => {
    vi.mocked(api.searchApi.search)
      .mockResolvedValueOnce(searchPage(['A'], { total: 20, total_exact: false, has_more: true }))
      .mockResolvedValueOnce(searchPage(['B'], { total: 15, total_exact: false, has_more: false }));
    await paging.startSearch({ query: 'q', mode: 'keyword', filters: {} }, quiet());
    expect(paging.totalText()).toBe('20+');
    await paging.loadMore();
    expect(paging.pageInfo().total).toBe(20);
    expect(paging.totalText()).toBe('≈20');
  });

  test('formatTotal: exact as is, inexact with more as a lower bound, else an estimate', () => {
    expect(paging.formatTotal(12, true, true)).toBe('12');
    expect(paging.formatTotal(12, false, true)).toBe('12+');
    expect(paging.formatTotal(12, false, false)).toBe('≈12');
  });

  test('a pending rerank holds loadMore; the reranked page 1 restarts the offset', async () => {
    const reranked = deferred();
    vi.mocked(api.searchApi.search)
      .mockResolvedValueOnce(searchPage(['F1', 'F2', 'F3'], { has_more: true }))
      .mockImplementationOnce(() => reranked.promise)
      .mockResolvedValueOnce(searchPage(['F1', 'N'], { has_more: false }));
    /** @type {any[]} */
    const seen = [];
    const run = paging.startSearch(
      { query: 'q', mode: 'hybrid', filters: {} },
      {
        capabilities: HYBRID,
        onStatus: () => {},
        onResults: (/** @type {any} */ r) => seen.push(r),
      }
    );
    await vi.waitFor(() => expect(seen.length).toBe(1));
    expect(paging.pageInfo().held).toBe(true);
    expect(await paging.loadMore(), 'held: nothing requested').toBeNull();
    expect(api.searchApi.search).toHaveBeenCalledTimes(2);

    reranked.resolve(searchPage(['R1', 'F1'], { has_more: true }));
    await run;
    expect(paging.pageInfo().held).toBe(false);
    expect(await paging.loadMore()).toEqual({ entries: rows(['N']) });
    const sent = vi.mocked(api.searchApi.search).mock.calls[2][0];
    expect(sent.offset, 'from the reranked page').toBe(2);
    expect(sent.advancedParams.rerank, 'a later page is never reranked').toBe(false);
  });
});

describe('reset and supersession', () => {
  test('reset aborts the request in flight and forgets the state', async () => {
    const pending = deferred();
    vi.mocked(api.entriesApi.list).mockImplementationOnce(() => pending.promise);
    const run = paging.startListing({});
    const signal = vi.mocked(api.entriesApi.list).mock.calls[0][0].signal;
    paging.reset();
    expect(signal.aborted).toBe(true);
    pending.resolve(listPage(['STALE'], 'C1'));
    expect(await run, 'a superseded response is never handed back').toBeNull();
    expect(paging.canLoadMore()).toBe(false);
    expect(paging.pageInfo().kind).toBeNull();
  });

  test('a new start aborts the previous run, whose response resolves to null', async () => {
    const pending = deferred();
    vi.mocked(api.entriesApi.list).mockImplementationOnce(() => pending.promise);
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(searchPage(['HIT']));
    const listing = paging.startListing({});
    const signal = vi.mocked(api.entriesApi.list).mock.calls[0][0].signal;
    await paging.startSearch({ query: 'q', mode: 'keyword', filters: {} }, quiet());
    expect(signal.aborted).toBe(true);
    pending.resolve(listPage(['STALE'], 'C1'));
    expect(await listing).toBeNull();
    expect(paging.pageInfo().kind).toBe('search');
  });

  test('an aborted request is not an error', async () => {
    vi.mocked(api.entriesApi.list).mockImplementationOnce(
      (/** @type {any} */ params) =>
        new Promise((_resolve, reject) => {
          params.signal.addEventListener('abort', () =>
            reject(new DOMException('aborted', 'AbortError'))
          );
        })
    );
    const run = paging.startListing({});
    paging.reset();
    await expect(run).resolves.toBeNull();
  });

  test('a real failure of the current run rejects and ends browse', async () => {
    vi.mocked(api.entriesApi.list).mockRejectedValueOnce(new Error('HTTP 503'));
    await expect(paging.startListing({})).rejects.toThrow('HTTP 503');
    expect(paging.canLoadMore()).toBe(false);
  });

  test('a loadMore superseded mid-flight resolves to null', async () => {
    const later = deferred();
    vi.mocked(api.entriesApi.list)
      .mockResolvedValueOnce(listPage(['A'], 'C1'))
      .mockImplementationOnce(() => later.promise);
    await paging.startListing({});
    const more = paging.loadMore();
    paging.reset();
    later.resolve(listPage(['B'], null));
    expect(await more).toBeNull();
  });
});

describe('listingFilter', () => {
  test('knob dimensions under the request filter; Mine replaces the author', () => {
    expect(
      paging.listingFilter({
        filters: { author: 'bob', similarity_threshold: 0.5 },
        filter: {},
        mailbox: 'all',
      })
    ).toEqual({ author: ['bob'] });
    expect(
      paging.listingFilter({
        filters: {},
        filter: { author: ['alice'], source_system: ['ops'] },
        mailbox: 'mine',
        mineAuthor: 'thorsten',
      })
    ).toEqual({ author: ['thorsten'], source_system: ['ops'] });
  });
});
