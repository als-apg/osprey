// @ts-check
/**
 * Front-end proofs for the two-phase reranked search (Task 4.3).
 *
 * Reranking costs seconds. Rather than hold a spinner for the whole of it, a
 * reranked search issues TWO queries: the fast ranking is fetched and painted
 * first with a "reranking…" status, then the reranked ranking replaces it
 * wholesale (the reranker draws from a larger candidate pool, so membership and
 * not just order may change). What these tests pin down:
 *
 *   - the exact wire bodies of both phases — the driver layers ONLY the
 *     per-phase `rerank` key on top of the operator's touched params, on a copy,
 *     and adds nothing else;
 *   - both paints happen, each carrying its own status line;
 *   - a superseded search's phase-2 response cannot repaint the view — the
 *     second query outlives the state that asked for it, so every render is
 *     guarded on still owning the newest search sequence;
 *   - a phase-2 failure keeps phase-1's results on screen behind a warning,
 *     covering BOTH shapes it arrives in: an HTTP rejection, and the backend's
 *     own 200-with-WARNING rerank fallback (indistinguishable from success
 *     except for the diagnostic);
 *   - Simple mode gets one plain-language line, and a UI-mode flip repaints the
 *     status the search actually finished on rather than resurrecting a stale
 *     "still reranking";
 *   - a mode with reranking off (or no reranker at all) still issues exactly
 *     ONE query, with no `rerank` key injected.
 *
 *   npx vitest run tests/interfaces/ariel/search-two-phase.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). Only api.js is mocked (the network
 * boundary); advanced-options.js, components.js and search.js all run for real,
 * so the effective-parameter read and the render sinks are exercised
 * end-to-end. search.js keeps its sequence/status at module scope, so every
 * test re-imports the graph fresh via vi.resetModules().
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
const OPTIONS_PATH = '../../../src/osprey/interfaces/ariel/static/js/advanced-options.js';

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
      modes: [
        { name: 'keyword', label: 'Keyword', description: 'Text search', parameters: [] },
      ],
    },
  },
  default_mode: 'keyword',
  shared_parameters: [],
  vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
});

/** The search view plus the advanced-options surface, as index.html lays them out. */
function mountFixture() {
  document.body.innerHTML = `
    <div id="search-mode-tabs" class="search-mode-tabs"></div>
    <div class="search-form" id="search-form">
      <div class="search-input-wrapper">
        <input type="text" id="search-input" class="input">
        <div class="search-input-actions">
          <button id="search-btn" class="btn btn-primary">Search</button>
        </div>
      </div>
    </div>
    <div class="search-options-bar">
      <button type="button" id="advanced-toggle-btn">Filters &amp; Options</button>
    </div>
    <div id="advanced-panel" class="advanced-panel hidden">
      <div class="advanced-header">
        <button type="button" id="advanced-reset-btn">Reset</button>
        <button type="button" id="advanced-close-btn">Close</button>
      </div>
      <div class="advanced-sections" id="advanced-sections"></div>
    </div>
    <div id="search-results"></div>
  `;
}

/**
 * Import the module graph fresh and initialise the panel from `capabilities`,
 * so `getEffectiveParam('rerank')` answers the way that deployment would.
 * @param {any} capabilities - /api/capabilities payload
 * @returns {Promise<{search: any, options: any, searchMock: any}>}
 */
async function loadModules(capabilities) {
  const api = await import(API_PATH);
  const options = await import(OPTIONS_PATH);
  const search = await import(SEARCH_PATH);
  options.initAdvancedOptions(capabilities);
  return { search, options, searchMock: vi.mocked(api.searchApi.search) };
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
  return new Promise(resolve => setTimeout(resolve, 0));
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
    author: 'thellert',
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

/** @returns {HTMLElement} */
function results() {
  return /** @type {HTMLElement} */ (document.getElementById('search-results'));
}

/** @returns {HTMLButtonElement} */
function searchBtn() {
  return /** @type {HTMLButtonElement} */ (document.getElementById('search-btn'));
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
  document.documentElement.removeAttribute('data-ui-mode');
  vi.restoreAllMocks();
});

describe('two-phase search: the wire bodies', () => {
  test('a reranked search issues exactly two queries, rerank false then true', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));

    await search.performSearch('quench');

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
    const { search, options, searchMock } = await loadModules(hybridCapabilities(true));
    // attachParamListeners only runs from renderAdvancedPanel, i.e. on open.
    /** @type {HTMLElement} */ (document.getElementById('advanced-toggle-btn')).click();
    const slider = /** @type {HTMLInputElement} */ (document.getElementById('param-candidate_limit'));
    slider.value = '80';
    slider.dispatchEvent(new Event('input'));
    expect(options.getAdvancedParams()).toEqual({ candidate_limit: 80 });

    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));
    await search.performSearch('quench');

    expect(searchMock.mock.calls[0][0].advancedParams)
      .toEqual({ candidate_limit: 80, rerank: false });
    expect(searchMock.mock.calls[1][0].advancedParams)
      .toEqual({ candidate_limit: 80, rerank: true });
    // The driver copied; it did not mutate the panel's own view of the world.
    expect(options.getAdvancedParams()).toEqual({ candidate_limit: 80 });
  });

  test('rerank off sends ONE query with no rerank key injected', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(false));
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));

    await search.performSearch('quench');

    expect(searchMock.mock.calls.length, 'single-phase path').toBe(1);
    expect(searchMock.mock.calls[0][0]).toEqual({
      query: 'quench',
      mode: 'hybrid',
      maxResults: 10,
      advancedParams: {},
    });
    expect(statusText(), 'no phase status on a single-phase search').toBeNull();
  });

  test('a mode with no reranker at all sends ONE query with no rerank key', async () => {
    const { search, searchMock } = await loadModules(KEYWORD_CAPABILITIES);
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));

    await search.performSearch('quench');

    expect(searchMock.mock.calls.length).toBe(1);
    expect(searchMock.mock.calls[0][0].mode).toBe('keyword');
    expect(searchMock.mock.calls[0][0].advancedParams).toEqual({});
    expect(statusText()).toBeNull();
  });
});

describe('two-phase search: both paints', () => {
  test('phase 1 paints the fast ranking, phase 2 replaces it', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    const reranked = deferred();
    searchMock.mockImplementationOnce(() => fast.promise).mockImplementationOnce(() => reranked.promise);

    const pending = search.performSearch('quench');

    fast.resolve(searchResponse({ entries: [entry('FAST-1'), entry('FAST-2')] }));
    await flush();

    expect(results().textContent).toContain('FAST-1');
    expect(statusText()).toBe('Search complete — reranking with LLM…');
    expect(searchBtn().disabled, 'still searching between the phases').toBe(true);

    // The reranked pool is not a re-ordering of the same set: membership moves.
    reranked.resolve(searchResponse({ entries: [entry('RERANKED-1'), entry('FAST-2')] }));
    await pending;

    expect(results().textContent).toContain('RERANKED-1');
    expect(results().textContent, 'phase 1 was replaced wholesale').not.toContain('FAST-1');
    expect(statusText()).toBe('Results updated after reranking');
    expect(searchBtn().disabled, 'controls handed back once both phases are done').toBe(false);
  });

  test('a search cannot start while the first one is still on its second phase', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    const reranked = deferred();
    searchMock.mockImplementationOnce(() => fast.promise).mockImplementationOnce(() => reranked.promise);

    const pending = search.performSearch('quench');
    fast.resolve(searchResponse({ entries: [entry('FAST-1')] }));
    await flush();

    await search.performSearch('second query');
    expect(searchMock.mock.calls.length, 'no third query was issued').toBe(2);

    reranked.resolve(searchResponse({ entries: [entry('RERANKED-1')] }));
    await pending;
  });
});

describe('two-phase search: a superseded phase 2 must not repaint', () => {
  test('clearing and re-searching strands the first search\'s phase 2', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    const fastA = deferred();
    const rerankedA = deferred();
    const fastB = deferred();
    searchMock
      .mockImplementationOnce(() => fastA.promise)
      .mockImplementationOnce(() => rerankedA.promise)
      .mockImplementationOnce(() => fastB.promise);

    const pendingA = search.performSearch('alpha');
    fastA.resolve(searchResponse({ entries: [entry('ALPHA-FAST')] }));
    await flush();
    expect(results().textContent).toContain('ALPHA-FAST');

    // The operator clears the box and searches again while alpha's reranked
    // query is still in flight.
    search.clearSearch();
    search.performSearch('beta');
    await flush();
    expect(searchMock.mock.calls.length, 'beta started its own phase 1').toBe(3);
    expect(searchMock.mock.calls[2][0].query).toBe('beta');

    rerankedA.resolve(searchResponse({ entries: [entry('ALPHA-RERANKED')] }));
    await pendingA;

    expect(results().textContent, 'the stranded response did not repaint')
      .not.toContain('ALPHA-RERANKED');
    expect(statusText(), 'nor did it plant a status line').toBeNull();
    expect(search.getSearchState().results, 'nor did it become the current results')
      .toBeNull();
    expect(search.getSearchState().isSearching, 'beta still owns the in-flight flag').toBe(true);
  });
});

describe('two-phase search: a phase-2 failure keeps the fast results', () => {
  test('an HTTP rejection leaves phase 1 on screen behind a warning', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    const reranked = deferred();
    searchMock.mockImplementationOnce(() => fast.promise).mockImplementationOnce(() => reranked.promise);

    const pending = search.performSearch('quench');
    fast.resolve(searchResponse({ entries: [entry('FAST-1'), entry('FAST-2')] }));
    await flush();

    reranked.reject(new Error('HTTP 504: reranker timed out loading the model'));
    await pending;

    expect(results().textContent, 'the fast ranking survived').toContain('FAST-1');
    expect(results().textContent).toContain('FAST-2');
    expect(statusText()).toBe('Could not improve the ranking — showing fast results');
    expect(results().querySelector('.text-error'), 'not an error state').toBeNull();
    expect(searchBtn().disabled, 'controls handed back').toBe(false);
    expect(search.getSearchState().results.entries[0].entry_id, 'phase 1 is still current')
      .toBe('FAST-1');
  });

  test('a 200 carrying the backend rerank warning reads as the fallback, not success', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    searchMock
      .mockResolvedValueOnce(searchResponse({ entries: [entry('FAST-1')] }))
      .mockResolvedValueOnce(searchResponse({
        entries: [entry('FAST-1')],
        diagnostics: [RERANK_FALLBACK_DIAGNOSTIC],
      }));

    await search.performSearch('quench');

    expect(statusText()).toBe('Could not improve the ranking — showing fast results');
  });

  test('an unrelated warning does not masquerade as the rerank fallback', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    searchMock
      .mockResolvedValueOnce(searchResponse({ entries: [entry('FAST-1')] }))
      .mockResolvedValueOnce(searchResponse({
        entries: [entry('RERANKED-1')],
        diagnostics: [{
          level: 'warning',
          source: 'vocabulary',
          category: 'expansion',
          message: 'One term could not be expanded',
        }],
      }));

    await search.performSearch('quench');

    expect(statusText()).toBe('Results updated after reranking');
  });

  test('a phase-1 failure is a failed search: no phase 2, error state', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    searchMock.mockRejectedValueOnce(new Error('HTTP 500: search backend down'));

    await search.performSearch('quench');

    expect(searchMock.mock.calls.length, 'no reranked query on a dead search').toBe(1);
    expect(results().querySelector('.text-error')).not.toBeNull();
    expect(statusText()).toBeNull();
    expect(searchBtn().disabled).toBe(false);
  });
});

describe('two-phase search: Simple mode', () => {
  test('Simple gets one plain-language line per phase', async () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    const reranked = deferred();
    searchMock.mockImplementationOnce(() => fast.promise).mockImplementationOnce(() => reranked.promise);

    const pending = search.performSearch('quench');
    fast.resolve(searchResponse({ entries: [entry('FAST-1')] }));
    await flush();

    expect(results().querySelectorAll('[data-testid="rerank-status"]').length, 'exactly one line')
      .toBe(1);
    expect(statusText()).toBe('Showing quick results — improving the order now…');
    // Simple deliberately omits the search-mechanics chrome, status line aside.
    expect(results().querySelector('.diagnostics-bar')).toBeNull();

    reranked.resolve(searchResponse({ entries: [entry('RERANKED-1')] }));
    await pending;

    expect(statusText()).toBe('Order improved — best matches first');
  });

  test('an empty fast ranking still gets its Simple status line', async () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    const reranked = deferred();
    searchMock.mockImplementationOnce(() => fast.promise).mockImplementationOnce(() => reranked.promise);

    const pending = search.performSearch('nothing matches this');
    fast.resolve(searchResponse({ entries: [] }));
    await flush();

    expect(results().textContent).toContain('No entries found');
    expect(statusText()).toBe('Showing quick results — improving the order now…');

    reranked.resolve(searchResponse({ entries: [entry('RERANKED-1')] }));
    await pending;
  });

  test('a UI-mode flip repaints the phase the search finished on, not a stale one', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    searchMock.mockResolvedValue(searchResponse({ entries: [entry('E1')] }));

    await search.performSearch('quench');
    expect(statusText()).toBe('Results updated after reranking');

    document.documentElement.setAttribute('data-ui-mode', 'simple');
    search.onUiModeChange();

    expect(statusText(), 'the finished phase, in Simple words')
      .toBe('Order improved — best matches first');
    expect(results().textContent, 'no resurrected "still reranking"')
      .not.toContain('improving the order now');
  });

  test('a flip mid-flight keeps saying reranking, because it still is', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    const reranked = deferred();
    searchMock.mockImplementationOnce(() => fast.promise).mockImplementationOnce(() => reranked.promise);

    const pending = search.performSearch('quench');
    fast.resolve(searchResponse({ entries: [entry('FAST-1')] }));
    await flush();
    expect(statusText()).toBe('Search complete — reranking with LLM…');

    document.documentElement.setAttribute('data-ui-mode', 'simple');
    search.onUiModeChange();
    expect(statusText()).toBe('Showing quick results — improving the order now…');

    reranked.resolve(searchResponse({ entries: [entry('RERANKED-1')] }));
    await pending;
    expect(statusText()).toBe('Order improved — best matches first');
  });
});

describe('two-phase search: the search button', () => {
  test('the button is disabled for the whole of a two-phase search', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    const reranked = deferred();
    searchMock.mockImplementationOnce(() => fast.promise).mockImplementationOnce(() => reranked.promise);

    expect(searchBtn().disabled).toBe(false);
    const pending = search.performSearch('quench');
    expect(searchBtn().disabled, 'disabled from the first request').toBe(true);

    fast.resolve(searchResponse({ entries: [entry('FAST-1')] }));
    await flush();
    expect(searchBtn().disabled, 'still disabled with phase-1 results on screen').toBe(true);

    reranked.resolve(searchResponse({ entries: [entry('RERANKED-1')] }));
    await pending;
    expect(searchBtn().disabled).toBe(false);
  });

  test('clearing an in-flight search hands the button back', async () => {
    const { search, searchMock } = await loadModules(hybridCapabilities(true));
    const fast = deferred();
    searchMock.mockImplementationOnce(() => fast.promise);

    search.performSearch('quench');
    expect(searchBtn().disabled).toBe(true);

    search.clearSearch();

    expect(searchBtn().disabled).toBe(false);
    expect(search.getSearchState().isSearching).toBe(false);
  });
});
