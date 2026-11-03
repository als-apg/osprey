// @ts-check
/**
 * Driver proofs for the two-phase reranked search.
 *
 * Reranking costs seconds. Rather than hold a spinner for the whole of it, a
 * reranked search issues TWO queries: the fast ranking is reported first with a
 * "reranking" status, then the reranked ranking replaces it wholesale (the
 * reranker draws from a larger candidate pool, so membership and not just order
 * may change). search.js is a DOM-free driver — runSearch(request, hooks) and
 * cancelSearch() — so these tests watch what it sends and what it reports
 * through onStatus/onResults. What they pin down:
 *
 *   - the exact wire bodies of both phases — the driver layers ONLY the
 *     per-phase `rerank` key on top of the sent params, on a copy, and adds
 *     nothing else;
 *   - the sent params are only the names the passed mode or the shared
 *     parameters declare, non-null, plus the caller's `author`;
 *   - a newer call or cancelSearch() supersedes an older search: its phase-2
 *     response reports nothing;
 *   - a phase-2 failure reports the fallback status and phase-1's results again,
 *     covering BOTH shapes it arrives in: an HTTP rejection, and the backend's
 *     own 200-with-WARNING rerank fallback;
 *   - a mode with reranking off (or no reranker at all) issues exactly ONE
 *     query, with no `rerank` key injected;
 *   - `rerank:'when-stable'` reranks only after a quiet second.
 *
 * The painting of all this is the list pane's job (list-paint.test.mjs).
 *
 *   npx vitest run tests/interfaces/ariel/search-two-phase.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). Only api.js is mocked (the network
 * boundary); advanced-options.js runs for real, so the effective-parameter read
 * is exercised end-to-end. search.js keeps its sequence at module scope, so
 * every test re-imports it fresh via vi.resetModules().
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    searchApi: { ...actual.searchApi, search: vi.fn() },
  };
});

const API_PATH = '../../../src/osprey/interfaces/ariel/static/js/api.js';
const SEARCH_PATH = '../../../src/osprey/interfaces/ariel/static/js/search.js';

/**
 * A capabilities payload whose single mode declares `rerank`.
 * @param {boolean} rerankDefault - The deployment's configured default
 * @returns {any}
 */
function hybridCapabilities(rerankDefault) {
  return {
    categories: {
      direct: {
        label: 'Direct',
        modes: [
          {
            name: 'hybrid',
            label: 'Hybrid',
            description: 'Keyword and semantic combined',
            parameters: [
              {
                name: 'rerank',
                label: 'Rerank Results',
                description: 'Re-score candidates with the cross-encoder',
                type: 'bool',
                default: rerankDefault,
                section: 'Retrieval',
              },
              {
                name: 'candidate_limit',
                label: 'Candidate Limit',
                description: 'How many candidates to retrieve before reranking',
                type: 'int',
                default: 40,
                min: 1,
                max: 200,
                step: 1,
                section: 'Retrieval',
              },
            ],
          },
        ],
      },
    },
    default_mode: 'hybrid',
    shared_parameters: [],
    vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
  };
}

/** A mode with no reranker at all: `rerank` is not in its descriptor list. */
const KEYWORD_CAPABILITIES = /** @type {any} */ ({
  categories: {
    direct: {
      label: 'Direct',
      modes: [{ name: 'keyword', label: 'Keyword', description: 'Text search', parameters: [] }],
    },
  },
  default_mode: 'keyword',
  shared_parameters: [],
  vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
});

/**
 * A promise whose settlement this test controls, so the DOM can be inspected
 * between phase 1 and phase 2.
 * @returns {{promise: Promise<any>, resolve: (value: any) => void, reject: (reason: any) => void}}
 */
function deferred() {
  /** @type {(value: any) => void} */
  let resolve = () => {};
  /** @type {(reason: any) => void} */
  let reject = () => {};
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Let every pending microtask (and the awaits chained behind it) run. */
function flush() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * A search response carrying the given entries and diagnostics.
 * @param {{entries?: any[], diagnostics?: any[]}} [overrides]
 * @returns {any}
 */
function searchResponse({ entries = [], diagnostics = [] } = {}) {
  return {
    answer: '',
    sources: [],
    search_modes_used: ['hybrid'],
    execution_time_ms: 5,
    total_results: entries.length,
    entries,
    diagnostics,
    expanded_terms: [],
  };
}

/**
 * @param {string} id - The entry id, used as the marker these tests look for
 * @returns {any} An entry shaped like the backend's
 */
function entry(id) {
  return {
    entry_id: id,
    timestamp: '2026-08-25T12:00:00Z',
    author: 'alice',
    source_system: 'demo',
    raw_text: `body of ${id}`,
    score: null,
    attachments: [],
    keywords: [],
    highlights: [],
  };
}

/** The backend's own rerank fallback: 200, fast ranking, warning diagnostic. */
const RERANK_FALLBACK_DIAGNOSTIC = {
  level: 'warning',
  source: 'hybrid',
  category: 'rerank',
  message: 'Reranker unavailable — returning the unreranked ranking',
};

/**
 * Import the driver fresh, with the search mock.
 * @returns {Promise<{search: typeof import('../../../src/osprey/interfaces/ariel/static/js/search.js'), searchMock: any}>}
 */
async function loadDriver() {
  const api = await import(API_PATH);
  const search = await import(SEARCH_PATH);
  return { search, searchMock: vi.mocked(api.searchApi.search) };
}

/**
 * Hooks that record every report, in order, as `['status', s]` / `['results', ids]`.
 * @param {any} capabilities
 * @param {'now'|'when-stable'} [rerank]
 */
function recorder(capabilities, rerank = 'now') {
  /** @type {any[]} */
  const log = [];
  return {
    log,
    hooks: {
      capabilities,
      rerank,
      /** @param {any} status */
      onStatus: (status) => log.push(['status', status]),
      /** @param {any} res */
      onResults: (res) =>
        log.push(['results', res.entries.map((/** @type {any} */ e) => e.entry_id)]),
    },
  };
}

/** A hybrid request with nothing touched. */
const HYBRID = { query: 'quench', mode: 'hybrid', filters: {} };

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('two-phase search: the wire bodies', () => {
  test('a reranked search issues exactly two queries, rerank false then true', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));

    await search.runSearch(HYBRID, recorder(hybridCapabilities(true)).hooks);

    expect(searchMock.mock.calls.length, 'one query per phase').toBe(2);
    expect(searchMock.mock.calls[0][0]).toEqual({
      query: 'quench',
      mode: 'hybrid',
      maxResults: 10,
      advancedParams: { rerank: false },
    });
    expect(searchMock.mock.calls[1][0]).toEqual({
      query: 'quench',
      mode: 'hybrid',
      maxResults: 10,
      advancedParams: { rerank: true },
    });
  });

  test('the touched params ride along untouched, with only rerank layered on', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));
    const filters = { candidate_limit: 80 };

    await search.runSearch({ ...HYBRID, filters }, recorder(hybridCapabilities(true)).hooks);

    expect(searchMock.mock.calls[0][0].advancedParams).toEqual({
      candidate_limit: 80,
      rerank: false,
    });
    expect(searchMock.mock.calls[1][0].advancedParams).toEqual({
      candidate_limit: 80,
      rerank: true,
    });
    // The driver copied; it did not mutate the caller's filters.
    expect(filters).toEqual({ candidate_limit: 80 });
  });

  test('rerank off sends ONE query with no rerank key injected', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));
    const { log, hooks } = recorder(hybridCapabilities(false));

    await search.runSearch(HYBRID, hooks);

    expect(searchMock.mock.calls.length, 'single-phase path').toBe(1);
    expect(searchMock.mock.calls[0][0]).toEqual({
      query: 'quench',
      mode: 'hybrid',
      maxResults: 10,
      advancedParams: {},
    });
    expect(log, 'no phase status on a single-phase search').toEqual([
      ['status', null],
      ['results', ['E1']],
    ]);
  });

  test('a touched rerank:false turns a reranking default into ONE query that says so', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));

    await search.runSearch(
      { ...HYBRID, filters: { rerank: false } },
      recorder(hybridCapabilities(true)).hooks
    );

    expect(searchMock.mock.calls.length).toBe(1);
    expect(searchMock.mock.calls[0][0].advancedParams).toEqual({ rerank: false });
  });

  test('a mode with no reranker at all sends ONE query with no rerank key', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));
    const { log, hooks } = recorder(KEYWORD_CAPABILITIES);

    await search.runSearch({ query: 'quench', mode: 'keyword', filters: {} }, hooks);

    expect(searchMock.mock.calls.length).toBe(1);
    expect(searchMock.mock.calls[0][0].mode).toBe('keyword');
    expect(searchMock.mock.calls[0][0].advancedParams).toEqual({});
    expect(log[0]).toEqual(['status', null]);
  });

  test('an empty query sends nothing and reports nothing', async () => {
    const { search, searchMock } = await loadDriver();
    const { log, hooks } = recorder(hybridCapabilities(true));

    await search.runSearch({ ...HYBRID, query: '   ' }, hooks);

    expect(searchMock).not.toHaveBeenCalled();
    expect(log).toEqual([]);
  });
});

describe('search driver: per-mode parameters', () => {
  /** Two modes plus a shared filter, so one mode's knob can leak into the other. */
  const TWO_MODES = /** @type {any} */ ({
    categories: {
      direct: {
        label: 'Direct',
        modes: [
          {
            name: 'keyword',
            label: 'Keyword',
            parameters: [
              { name: 'fuzzy', label: 'Fuzzy', type: 'bool', default: false },
              { name: 'max_results', label: 'Max', type: 'int', default: 10 },
            ],
          },
          {
            name: 'semantic',
            label: 'Semantic',
            parameters: [{ name: 'similarity', label: 'Similarity', type: 'float', default: 0.5 }],
          },
        ],
      },
    },
    shared_parameters: [{ name: 'author', label: 'Author', type: 'text', default: null }],
  });

  test("another mode's touched knob stays off the wire", async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse());

    await search.runSearch(
      { query: 'q', mode: 'semantic', filters: { fuzzy: true, similarity: 0.8 } },
      recorder(TWO_MODES).hooks
    );

    expect(searchMock.mock.calls[0][0].advancedParams).toEqual({ similarity: 0.8 });
  });

  test('shared parameters are sent in every mode; null values never are', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse());

    await search.runSearch(
      { query: 'q', mode: 'keyword', filters: { author: 'alice', fuzzy: null } },
      recorder(TWO_MODES).hooks
    );

    expect(searchMock.mock.calls[0][0].advancedParams).toEqual({ author: 'alice' });
  });

  test("the caller's author is sent and wins over an author filter", async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse());

    await search.runSearch(
      { query: 'q', mode: 'semantic', filters: { author: 'alice' }, author: 'bob' },
      recorder(TWO_MODES).hooks
    );

    expect(searchMock.mock.calls[0][0].advancedParams).toEqual({ author: 'bob' });
  });

  test('a touched max_results sets the request size', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse());

    await search.runSearch(
      { query: 'q', mode: 'keyword', filters: { max_results: 25 } },
      recorder(TWO_MODES).hooks
    );

    expect(searchMock.mock.calls[0][0].maxResults).toBe(25);
  });
});

describe('two-phase search: both reports', () => {
  test('phase 1 reports the fast ranking, phase 2 replaces it', async () => {
    const { search, searchMock } = await loadDriver();
    const fast = deferred();
    const reranked = deferred();
    searchMock
      .mockImplementationOnce(() => fast.promise)
      .mockImplementationOnce(() => reranked.promise);
    const { log, hooks } = recorder(hybridCapabilities(true));

    const pending = search.runSearch(HYBRID, hooks);

    fast.resolve(searchResponse({ entries: [entry('FAST-1'), entry('FAST-2')] }));
    await flush();
    expect(log).toEqual([
      ['status', 'reranking'],
      ['results', ['FAST-1', 'FAST-2']],
    ]);

    reranked.resolve(searchResponse({ entries: [entry('RERANKED-1'), entry('FAST-2')] }));
    await pending;

    expect(log.slice(2)).toEqual([
      ['status', 'updated'],
      ['results', ['RERANKED-1', 'FAST-2']],
    ]);
  });

  test('a newer call supersedes a search that is still on its second phase', async () => {
    const { search, searchMock } = await loadDriver();
    const fast = deferred();
    const reranked = deferred();
    const fastB = deferred();
    searchMock
      .mockImplementationOnce(() => fast.promise)
      .mockImplementationOnce(() => reranked.promise)
      .mockImplementationOnce(() => fastB.promise);
    const first = recorder(hybridCapabilities(true));
    const second = recorder(hybridCapabilities(true));

    const pending = search.runSearch(HYBRID, first.hooks);
    fast.resolve(searchResponse({ entries: [entry('FAST-1')] }));
    await flush();

    search.runSearch({ ...HYBRID, query: 'second query' }, second.hooks);
    await flush();
    expect(searchMock.mock.calls.length, 'the newer call issued its own phase 1').toBe(3);
    expect(searchMock.mock.calls[2][0].query).toBe('second query');

    reranked.resolve(searchResponse({ entries: [entry('RERANKED-1')] }));
    await pending;

    expect(first.log, "the superseded search's phase 2 reported nothing").toEqual([
      ['status', 'reranking'],
      ['results', ['FAST-1']],
    ]);
    expect(second.log).toEqual([]);
  });
});

describe('two-phase search: a superseded phase 2 must not report', () => {
  test('cancelSearch strands an in-flight phase 2', async () => {
    const { search, searchMock } = await loadDriver();
    const fast = deferred();
    const reranked = deferred();
    searchMock
      .mockImplementationOnce(() => fast.promise)
      .mockImplementationOnce(() => reranked.promise);
    const { log, hooks } = recorder(hybridCapabilities(true));

    const pending = search.runSearch(HYBRID, hooks);
    fast.resolve(searchResponse({ entries: [entry('ALPHA-FAST')] }));
    await flush();

    search.cancelSearch();
    reranked.resolve(searchResponse({ entries: [entry('ALPHA-RERANKED')] }));
    await pending;

    expect(log).toEqual([
      ['status', 'reranking'],
      ['results', ['ALPHA-FAST']],
    ]);
  });

  test('a cancelled search whose phase 1 fails neither reports nor rejects', async () => {
    const { search, searchMock } = await loadDriver();
    const fast = deferred();
    searchMock.mockImplementationOnce(() => fast.promise);
    const { log, hooks } = recorder(hybridCapabilities(true));

    const pending = search.runSearch(HYBRID, hooks);
    search.cancelSearch();
    fast.reject(new Error('HTTP 500'));

    await expect(pending).resolves.toBeUndefined();
    expect(log).toEqual([]);
    expect(searchMock.mock.calls.length).toBe(1);
  });
});

describe('two-phase search: a phase-2 failure keeps the fast results', () => {
  test('an HTTP rejection reports the fallback with phase 1 results', async () => {
    const { search, searchMock } = await loadDriver();
    const fast = deferred();
    const reranked = deferred();
    searchMock
      .mockImplementationOnce(() => fast.promise)
      .mockImplementationOnce(() => reranked.promise);
    const { log, hooks } = recorder(hybridCapabilities(true));

    const pending = search.runSearch(HYBRID, hooks);
    fast.resolve(searchResponse({ entries: [entry('FAST-1'), entry('FAST-2')] }));
    await flush();

    reranked.reject(new Error('HTTP 504: reranker timed out loading the model'));
    await expect(pending, 'not a failed search').resolves.toBeUndefined();

    expect(log.slice(2)).toEqual([
      ['status', 'fallback'],
      ['results', ['FAST-1', 'FAST-2']],
    ]);
  });

  test('a 200 carrying the backend rerank warning reads as the fallback, not success', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock
      .mockResolvedValueOnce(searchResponse({ entries: [entry('FAST-1')] }))
      .mockResolvedValueOnce(
        searchResponse({
          entries: [entry('FAST-1')],
          diagnostics: [RERANK_FALLBACK_DIAGNOSTIC],
        })
      );
    const { log, hooks } = recorder(hybridCapabilities(true));

    await search.runSearch(HYBRID, hooks);

    expect(log[2]).toEqual(['status', 'fallback']);
  });

  test('an unrelated warning does not masquerade as the rerank fallback', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock
      .mockResolvedValueOnce(searchResponse({ entries: [entry('FAST-1')] }))
      .mockResolvedValueOnce(
        searchResponse({
          entries: [entry('RERANKED-1')],
          diagnostics: [
            {
              level: 'warning',
              source: 'vocabulary',
              category: 'expansion',
              message: 'One term could not be expanded',
            },
          ],
        })
      );
    const { log, hooks } = recorder(hybridCapabilities(true));

    await search.runSearch(HYBRID, hooks);

    expect(log[2]).toEqual(['status', 'updated']);
  });

  test('a phase-1 failure is a failed search: no phase 2, the promise rejects', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockRejectedValueOnce(new Error('HTTP 500: search backend down'));
    const { log, hooks } = recorder(hybridCapabilities(true));

    await expect(search.runSearch(HYBRID, hooks)).rejects.toThrow('search backend down');

    expect(searchMock.mock.calls.length, 'no reranked query on a dead search').toBe(1);
    expect(log).toEqual([]);
  });
});

describe("two-phase search: rerank 'when-stable'", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  });

  test('phase 1 runs at once; phase 2 only after a quiet second', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock
      .mockResolvedValueOnce(searchResponse({ entries: [entry('FAST-1')] }))
      .mockResolvedValueOnce(searchResponse({ entries: [entry('RERANKED-1')] }));
    const { log, hooks } = recorder(hybridCapabilities(true), 'when-stable');

    const pending = search.runSearch(HYBRID, hooks);
    await vi.advanceTimersByTimeAsync(0);
    expect(searchMock.mock.calls.length, 'phase 1 at once').toBe(1);
    expect(log).toEqual([
      ['status', 'reranking'],
      ['results', ['FAST-1']],
    ]);

    await vi.advanceTimersByTimeAsync(search.RERANK_SETTLE_MS - 1);
    expect(searchMock.mock.calls.length, 'no rerank inside the window').toBe(1);

    await vi.advanceTimersByTimeAsync(1);
    await pending;
    expect(searchMock.mock.calls.length).toBe(2);
    expect(searchMock.mock.calls[1][0].advancedParams).toEqual({ rerank: true });
    expect(log.slice(2)).toEqual([
      ['status', 'updated'],
      ['results', ['RERANKED-1']],
    ]);
  });

  test('a newer call inside the window means the older search never reranks', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockImplementation(async (/** @type {any} */ req) =>
      searchResponse({ entries: [entry(`${req.query}-${req.advancedParams.rerank}`)] })
    );
    const first = recorder(hybridCapabilities(true), 'when-stable');
    const second = recorder(hybridCapabilities(true), 'when-stable');

    const pendingA = search.runSearch({ ...HYBRID, query: 'que' }, first.hooks);
    await vi.advanceTimersByTimeAsync(500);
    const pendingB = search.runSearch({ ...HYBRID, query: 'quench' }, second.hooks);
    await pendingA;
    await vi.advanceTimersByTimeAsync(search.RERANK_SETTLE_MS);
    await pendingB;

    const queries = searchMock.mock.calls.map((/** @type {any} */ c) => [
      c[0].query,
      c[0].advancedParams.rerank,
    ]);
    expect(queries).toEqual([
      ['que', false],
      ['quench', false],
      ['quench', true],
    ]);
    expect(first.log).toEqual([
      ['status', 'reranking'],
      ['results', ['que-false']],
    ]);
    expect(second.log.slice(2)).toEqual([
      ['status', 'updated'],
      ['results', ['quench-true']],
    ]);
  });

  test("rerank 'now' does not wait", async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));

    await search.runSearch(HYBRID, recorder(hybridCapabilities(true), 'now').hooks);

    expect(searchMock.mock.calls.length).toBe(2);
  });
});

describe('search driver: cancelSearch', () => {
  test('cancelling inside the when-stable window releases the search at once, without phase 2', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('FAST-1')] }));
    const { log, hooks } = recorder(hybridCapabilities(true), 'when-stable');

    let settled = false;
    const pending = search.runSearch(HYBRID, hooks).then(() => {
      settled = true;
    });
    await vi.advanceTimersByTimeAsync(0);

    search.cancelSearch();
    await vi.advanceTimersByTimeAsync(0);
    expect(settled, 'released without waiting out the window').toBe(true);
    await pending;

    await vi.advanceTimersByTimeAsync(search.RERANK_SETTLE_MS * 2);
    expect(searchMock.mock.calls.length, 'no phase 2').toBe(1);
    expect(log).toEqual([
      ['status', 'reranking'],
      ['results', ['FAST-1']],
    ]);
  });

  test('a search started after cancelSearch runs normally', async () => {
    const { search, searchMock } = await loadDriver();
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));
    search.cancelSearch();
    const { log, hooks } = recorder(hybridCapabilities(false));

    await search.runSearch(HYBRID, hooks);

    expect(log).toEqual([
      ['status', null],
      ['results', ['E1']],
    ]);
  });
});
