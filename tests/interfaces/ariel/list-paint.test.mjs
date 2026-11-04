// @ts-check
/**
 * Paint proofs for the two-phase reranked search, driven through the list
 * pane (list.js) the way the operator drives it: a query written to the store.
 *
 * These cases came from search-two-phase.test.mjs, where they drove the old
 * performSearch/clearSearch and its #search-results render. search.js is now a
 * DOM-free driver, so what is proved here is the painting: both phases paint
 * with their status line, a superseded phase 2 does not repaint, a phase-2
 * failure keeps the fast results behind a warning, a phase-1 failure paints the
 * error state, and the list stays `aria-busy` across both phases. A newer query
 * now supersedes a search still on its second phase (the old "cannot start"
 * rule is inverted). The status strings they pin:
 *
 *   reranking: 'Search complete — reranking with LLM…'
 *   updated:   'Results updated after reranking'
 *   fallback:  'Could not improve the ranking — showing fast results'
 *
 *   npx vitest run tests/interfaces/ariel/list-paint.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    searchApi: { ...actual.searchApi, search: vi.fn() },
    entriesApi: { ...actual.entriesApi, list: vi.fn(), get: vi.fn() },
  };
});

const API_PATH = '../../../src/osprey/interfaces/ariel/static/js/api.js';
const STORE_PATH = '../../../src/osprey/interfaces/ariel/static/js/store.js';
const LIST_PATH = '../../../src/osprey/interfaces/ariel/static/js/list.js';

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

/** The list pane root, as index.html provides it. */
function mountFixture() {
  document.body.innerHTML = '<section id="ariel-list"></section>';
}

/**
 * Import the module graph fresh, mount the list pane from `capabilities` and
 * return the store plus the search mock. The store and list must come from the
 * same module reset, so the list subscribes to the store this test drives.
 * @param {any} capabilities - /api/capabilities payload
 * @returns {Promise<{store: any, list: any, searchMock: any}>}
 */
async function loadModules(capabilities) {
  const api = await import(API_PATH);
  const store = await import(STORE_PATH);
  const list = await import(LIST_PATH);
  vi.mocked(api.entriesApi.list).mockResolvedValue({
    entries: [],
    total: 0,
    page: 1,
    total_pages: 1,
  });
  list.initList(/** @type {HTMLElement} */ (document.getElementById('ariel-list')), {
    capabilities,
  });
  await flush();
  vi.mocked(api.entriesApi.list).mockClear();
  return { store, list, searchMock: vi.mocked(api.searchApi.search) };
}

/**
 * Type a query into the store the way the search box's Enter does.
 * @param {any} store
 * @param {string} query
 */
function search(store, query) {
  store.setState({ query, rerank: 'now' }, { force: true });
}

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
    subject: null,
    snippet: `body of ${id}`,
    score: null,
    attachment_count: 0,
    enhancements: {},
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

/** @returns {HTMLElement} */
function results() {
  return /** @type {HTMLElement} */ (document.getElementById('ariel-list-body'));
}

/**
 * Whether the list says a search is still running.
 * @returns {boolean}
 */
function busy() {
  return results().getAttribute('aria-busy') === 'true';
}

/**
 * Wait for the driver's awaited promise to settle and the list to paint.
 */
async function settled() {
  await flush();
  await flush();
}

/**
 * The rerank status line's collapsed text, or null when no line is rendered.
 * @returns {string|null}
 */
function statusText() {
  const el = results().querySelector('[data-testid="rerank-status"]');
  return el ? (el.textContent ?? '').replace(/\s+/g, ' ').trim() : null;
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  mountFixture();
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('two-phase search: both paints', () => {
  test('phase 1 paints the fast ranking, phase 2 replaces it', async () => {
    const { store, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    const reranked = deferred();
    searchMock
      .mockImplementationOnce(() => fast.promise)
      .mockImplementationOnce(() => reranked.promise);

    search(store, 'quench');

    fast.resolve(searchResponse({ entries: [entry('FAST-1'), entry('FAST-2')] }));
    await flush();

    expect(results().textContent).toContain('FAST-1');
    expect(statusText()).toBe('Search complete — reranking with LLM…');
    expect(busy(), 'still searching between the phases').toBe(true);

    // The reranked pool is not a re-ordering of the same set: membership moves.
    reranked.resolve(searchResponse({ entries: [entry('RERANKED-1'), entry('FAST-2')] }));
    await settled();

    expect(results().textContent).toContain('RERANKED-1');
    expect(results().textContent, 'phase 1 was replaced wholesale').not.toContain('FAST-1');
    expect(statusText()).toBe('Results updated after reranking');
    expect(busy(), 'busy cleared once both phases are done').toBe(false);
  });

  test('a newer query during phase 2 supersedes the first search', async () => {
    // Inverted from "a search cannot start while the first one is still on its
    // second phase": the driver now lets a newer call win.
    const { store, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    const reranked = deferred();
    searchMock
      .mockImplementationOnce(() => fast.promise)
      .mockImplementationOnce(() => reranked.promise)
      .mockResolvedValueOnce(searchResponse({ entries: [entry('SECOND-FAST')] }))
      .mockResolvedValueOnce(searchResponse({ entries: [entry('SECOND-RERANKED')] }));

    search(store, 'quench');
    fast.resolve(searchResponse({ entries: [entry('FAST-1')] }));
    await flush();

    search(store, 'second query');
    await settled();
    expect(searchMock.mock.calls.length, 'the second query was issued').toBe(4);
    expect(searchMock.mock.calls[2][0].query).toBe('second query');

    reranked.resolve(searchResponse({ entries: [entry('RERANKED-1')] }));
    await settled();
    expect(results().textContent).toContain('SECOND-RERANKED');
    expect(results().textContent, 'the superseded phase 2 did not repaint').not.toContain(
      'RERANKED-1'
    );
  });
});

describe('two-phase search: a superseded phase 2 must not repaint', () => {
  test("clearing and re-searching strands the first search's phase 2", async () => {
    const { store, searchMock } = await loadModules(hybridCapabilities(true));
    const fastA = deferred();
    const rerankedA = deferred();
    const fastB = deferred();
    searchMock
      .mockImplementationOnce(() => fastA.promise)
      .mockImplementationOnce(() => rerankedA.promise)
      .mockImplementationOnce(() => fastB.promise);

    search(store, 'alpha');
    fastA.resolve(searchResponse({ entries: [entry('ALPHA-FAST')] }));
    await flush();
    expect(results().textContent).toContain('ALPHA-FAST');

    // The operator clears the box and searches again while alpha's reranked
    // query is still in flight.
    search(store, '');
    search(store, 'beta');
    await flush();
    expect(searchMock.mock.calls.length, 'beta started its own phase 1').toBe(3);
    expect(searchMock.mock.calls[2][0].query).toBe('beta');

    rerankedA.resolve(searchResponse({ entries: [entry('ALPHA-RERANKED')] }));
    await settled();

    expect(results().textContent, 'the stranded response did not repaint').not.toContain(
      'ALPHA-RERANKED'
    );
    expect(statusText(), 'nor did it plant a status line').toBeNull();
    expect(busy(), 'beta still owns the busy flag').toBe(true);
  });
});

describe('two-phase search: a phase-2 failure keeps the fast results', () => {
  test('an HTTP rejection leaves phase 1 on screen behind a warning', async () => {
    const { store, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    const reranked = deferred();
    searchMock
      .mockImplementationOnce(() => fast.promise)
      .mockImplementationOnce(() => reranked.promise);

    search(store, 'quench');
    fast.resolve(searchResponse({ entries: [entry('FAST-1'), entry('FAST-2')] }));
    await flush();

    reranked.reject(new Error('HTTP 504: reranker timed out loading the model'));
    await settled();

    expect(results().textContent, 'the fast ranking survived').toContain('FAST-1');
    expect(results().textContent).toContain('FAST-2');
    expect(statusText()).toBe('Could not improve the ranking — showing fast results');
    expect(results().querySelector('.text-error'), 'not an error state').toBeNull();
    expect(busy(), 'busy cleared').toBe(false);
    expect(
      /** @type {HTMLElement} */ (results().querySelector('[data-entry-id]')).dataset.entryId,
      'phase 1 is still current'
    ).toBe('FAST-1');
  });

  test('a 200 carrying the backend rerank warning reads as the fallback, not success', async () => {
    const { store, searchMock } = await loadModules(hybridCapabilities(true));
    searchMock
      .mockResolvedValueOnce(searchResponse({ entries: [entry('FAST-1')] }))
      .mockResolvedValueOnce(
        searchResponse({
          entries: [entry('FAST-1')],
          diagnostics: [RERANK_FALLBACK_DIAGNOSTIC],
        })
      );

    search(store, 'quench');
    await settled();

    expect(statusText()).toBe('Could not improve the ranking — showing fast results');
  });

  test('an unrelated warning does not masquerade as the rerank fallback', async () => {
    const { store, searchMock } = await loadModules(hybridCapabilities(true));
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

    search(store, 'quench');
    await settled();

    expect(statusText()).toBe('Results updated after reranking');
  });

  test('a phase-1 failure is a failed search: no phase 2, error state', async () => {
    const { store, searchMock } = await loadModules(hybridCapabilities(true));
    searchMock.mockRejectedValueOnce(new Error('HTTP 500: search backend down'));

    search(store, 'quench');
    await settled();

    expect(searchMock.mock.calls.length, 'no reranked query on a dead search').toBe(1);
    expect(results().querySelector('.text-error')).not.toBeNull();
    expect(statusText()).toBeNull();
    expect(busy()).toBe(false);
  });
});

describe('two-phase search: the busy flag', () => {
  test('the list is busy for the whole of a two-phase search', async () => {
    const { store, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    const reranked = deferred();
    searchMock
      .mockImplementationOnce(() => fast.promise)
      .mockImplementationOnce(() => reranked.promise);

    expect(busy()).toBe(false);
    search(store, 'quench');
    expect(busy(), 'busy from the first request').toBe(true);

    fast.resolve(searchResponse({ entries: [entry('FAST-1')] }));
    await flush();
    expect(busy(), 'still busy with phase-1 results on screen').toBe(true);

    reranked.resolve(searchResponse({ entries: [entry('RERANKED-1')] }));
    await settled();
    expect(busy()).toBe(false);
  });

  test('clearing an in-flight search hands the list back to the listing', async () => {
    const { store, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    searchMock.mockImplementationOnce(() => fast.promise);

    search(store, 'quench');
    expect(busy()).toBe(true);

    search(store, '');
    await settled();

    expect(busy()).toBe(false);
    fast.resolve(searchResponse({ entries: [entry('LATE-FAST')] }));
    await settled();
    expect(results().textContent, 'the cleared search never paints').not.toContain('LATE-FAST');
  });
});

describe('search issues on the mode chip', () => {
  const PICTURE_DIAGNOSTIC = {
    level: 'warning',
    source: 'hybrid',
    message: 'Picture search unavailable',
  };

  /** @returns {HTMLButtonElement} */
  function chip() {
    return /** @type {HTMLButtonElement} */ (document.getElementById('ariel-mode-chip'));
  }

  test('a diagnostic is a dot on the chip, not a banner over the results', async () => {
    const { store, searchMock } = await loadModules(hybridCapabilities(false));
    searchMock.mockResolvedValue(
      searchResponse({ entries: [entry('E1')], diagnostics: [PICTURE_DIAGNOSTIC] })
    );

    search(store, 'quench');
    await settled();

    expect(results().querySelector('.diagnostics-bar'), 'no banner in the list').toBeNull();
    const dot = chip().querySelector('.ariel-mode-chip-dot');
    expect(dot, 'the chip carries the dot').not.toBeNull();
    expect(dot?.getAttribute('data-level')).toBe('warning');
    expect(chip().getAttribute('aria-description')).toContain('Picture search unavailable');
    expect(chip().title).toContain('Picture search unavailable');
  });

  test('the chip popover lists the issue, so a keyboard user reaches it', async () => {
    const { store, searchMock } = await loadModules(hybridCapabilities(false));
    searchMock.mockResolvedValue(
      searchResponse({ entries: [entry('E1')], diagnostics: [PICTURE_DIAGNOSTIC] })
    );

    search(store, 'quench');
    await settled();

    const issues = /** @type {HTMLElement} */ (
      document.querySelector('#ariel-filters-popover [data-search-issues]')
    );
    expect(issues.hidden).toBe(false);
    expect(issues.textContent).toContain('Picture search unavailable');
  });

  test('a clean search clears the dot and the popover issues', async () => {
    const { store, searchMock } = await loadModules(hybridCapabilities(false));
    searchMock
      .mockResolvedValueOnce(
        searchResponse({ entries: [entry('E1')], diagnostics: [PICTURE_DIAGNOSTIC] })
      )
      .mockResolvedValueOnce(searchResponse({ entries: [entry('E2')] }));

    search(store, 'quench');
    await settled();
    search(store, 'vacuum');
    await settled();

    expect(chip().querySelector('.ariel-mode-chip-dot')).toBeNull();
    expect(chip().hasAttribute('aria-description')).toBe(false);
    const issues = /** @type {HTMLElement} */ (
      document.querySelector('#ariel-filters-popover [data-search-issues]')
    );
    expect(issues.hidden).toBe(true);
  });
});
