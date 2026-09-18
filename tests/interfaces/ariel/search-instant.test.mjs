// @ts-check
/**
 * Front-end proofs for search-as-you-type.
 *
 * A mode whose capabilities declare `instant_search` runs on every keystroke
 * instead of on Enter. That is only affordable because of three guards, and
 * only correct because of a fourth, and this file pins down all four:
 *
 *   - a burst of typing is debounced into ONE request rather than one per
 *     character;
 *   - a query shorter than the minimum is never sent, and clearing the input
 *     clears the results rather than leaving the previous query's on screen;
 *   - requests over the in-flight cap are DROPPED, not queued — a queued
 *     request answers a query the operator has already left;
 *   - a slower response that is no longer the newest search cannot repaint,
 *     and a full search (Enter) supersedes every instant search in flight and
 *     keeps its own control of the search button.
 *
 * A mode that declares no `instant_search` parameter — every built-in one —
 * must be untouched by any of this: typing into it issues nothing at all.
 *
 *   npx vitest run tests/interfaces/ariel/search-instant.test.mjs
 *
 * Runs under happy-dom. Only api.js is mocked (the network boundary);
 * advanced-options.js and search.js run for real, so the effective-parameter
 * read is exercised rather than stubbed. search.js keeps its sequence and
 * in-flight counters at module scope, so every test re-imports the graph fresh
 * via vi.resetModules().
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

/** The debounce search.js applies, plus a margin. */
const PAST_DEBOUNCE_MS = 200;

/**
 * A capabilities payload whose single mode declares `instant_search`.
 * @param {boolean} instantDefault - The deployment's configured default
 * @returns {any}
 */
function jevCapabilities(instantDefault) {
  return {
    categories: {
      direct: {
        label: 'Direct',
        modes: [
          {
            name: 'jev',
            label: 'Jev',
            description: 'Keyword retrieval reranked by Jev',
            parameters: [
              {
                name: 'instant_search',
                label: 'Search as you type',
                description: 'Run the search on every keystroke',
                type: 'bool',
                default: instantDefault,
                section: 'Retrieval',
              },
            ],
          },
        ],
      },
    },
    default_mode: 'jev',
    shared_parameters: [],
    vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
  };
}

/** A mode with no instant search at all: the parameter is not in its list. */
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

/** The search view, as index.html lays it out. */
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
      <div class="advanced-sections" id="advanced-sections"></div>
    </div>
    <div id="search-results"></div>
  `;
}

/**
 * Import the module graph fresh, initialise the panel, and wire the listeners.
 * @param {any} capabilities - /api/capabilities payload
 * @returns {Promise<{search: any, searchMock: any}>}
 */
async function loadModules(capabilities) {
  const api = await import(API_PATH);
  const options = await import(OPTIONS_PATH);
  const search = await import(SEARCH_PATH);
  options.initAdvancedOptions(capabilities);
  search.initSearch(capabilities);
  return { search, searchMock: vi.mocked(api.searchApi.search) };
}

/**
 * Type into the search input, firing the `input` event the listener is on.
 * @param {string} value - The input's new value
 */
function type(value) {
  const input = /** @type {HTMLInputElement} */ (document.getElementById('search-input'));
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

/** Press Enter in the search input. */
function pressEnter() {
  const input = /** @type {HTMLInputElement} */ (document.getElementById('search-input'));
  input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
}

/**
 * A promise whose settlement this test controls.
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
 * A search response carrying the given entries.
 * @param {string[]} ids - Entry ids to return
 * @returns {any}
 */
function searchResponse(ids) {
  return {
    answer: '',
    sources: [],
    search_modes_used: ['jev'],
    execution_time_ms: 5,
    total_results: ids.length,
    entries: ids.map(id => ({
      entry_id: id,
      timestamp: '2026-08-25T12:00:00Z',
      author: 'alice',
      source_system: 'demo',
      raw_text: `body of ${id}`,
      score: null,
      attachments: [],
      keywords: [],
      highlights: [],
    })),
    diagnostics: [],
    expanded_terms: [],
  };
}

/** @returns {HTMLElement} */
function results() {
  return /** @type {HTMLElement} */ (document.getElementById('search-results'));
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  vi.useFakeTimers({ shouldAdvanceTime: true });
  mountFixture();
});

afterEach(() => {
  vi.useRealTimers();
  document.body.innerHTML = '';
});

describe('opting in', () => {
  test('a mode without the parameter issues nothing when the operator types', async () => {
    const { searchMock } = await loadModules(KEYWORD_CAPABILITIES);

    type('beam loss');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);

    expect(searchMock).not.toHaveBeenCalled();
  });

  test('a mode with the parameter off issues nothing either', async () => {
    const { searchMock } = await loadModules(jevCapabilities(false));

    type('beam loss');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);

    expect(searchMock).not.toHaveBeenCalled();
  });

  test('a mode with the parameter on searches without Enter', async () => {
    const { searchMock } = await loadModules(jevCapabilities(true));
    searchMock.mockResolvedValue(searchResponse(['e1']));

    type('beam loss');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);
    await flush();

    expect(searchMock).toHaveBeenCalledTimes(1);
    expect(searchMock.mock.calls[0][0]).toMatchObject({ query: 'beam loss', mode: 'jev' });
    expect(results().textContent).toContain('e1');
  });
});

describe('what never reaches the wire', () => {
  test('a burst of typing collapses into one request', async () => {
    const { searchMock } = await loadModules(jevCapabilities(true));
    searchMock.mockResolvedValue(searchResponse(['e1']));

    for (const value of ['be', 'bea', 'beam', 'beam ', 'beam l']) {
      type(value);
      await vi.advanceTimersByTimeAsync(20);
    }
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);
    await flush();

    expect(searchMock).toHaveBeenCalledTimes(1);
    expect(searchMock.mock.calls[0][0].query).toBe('beam l');
  });

  test('a one-character query is too short to be worth a request', async () => {
    const { searchMock } = await loadModules(jevCapabilities(true));

    type('b');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);

    expect(searchMock).not.toHaveBeenCalled();
  });

  test('clearing the input clears the results rather than leaving stale ones', async () => {
    const { searchMock } = await loadModules(jevCapabilities(true));
    searchMock.mockResolvedValue(searchResponse(['e1']));

    type('beam loss');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);
    await flush();
    expect(results().textContent).toContain('e1');

    type('');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);

    expect(results().innerHTML).toBe('');
    expect(searchMock).toHaveBeenCalledTimes(1);
  });

  test('a keystroke over the in-flight cap is dropped, not queued', async () => {
    const { searchMock } = await loadModules(jevCapabilities(true));
    const pending = [deferred(), deferred(), deferred(), deferred()];
    let issued = 0;
    searchMock.mockImplementation(() => {
      const next = pending[issued];
      issued++;
      return next ? next.promise : Promise.resolve(searchResponse([]));
    });

    // Five keystrokes, each past the debounce, none of them answered yet.
    for (const value of ['aa', 'aab', 'aabc', 'aabcd', 'aabcde']) {
      type(value);
      await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);
      await flush();
    }

    // Four in flight is the cap; the fifth never left.
    expect(searchMock).toHaveBeenCalledTimes(4);

    pending.forEach(p => p.resolve(searchResponse([])));
    await flush();
  });
});

describe('staleness', () => {
  test('a slower earlier response cannot repaint over a newer one', async () => {
    const { searchMock } = await loadModules(jevCapabilities(true));
    const slow = deferred();
    const fast = deferred();
    searchMock.mockReturnValueOnce(slow.promise).mockReturnValueOnce(fast.promise);

    type('beam');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);
    await flush();
    type('beam loss');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);
    await flush();

    fast.resolve(searchResponse(['newest']));
    await flush();
    slow.resolve(searchResponse(['stale']));
    await flush();

    expect(results().textContent).toContain('newest');
    expect(results().textContent).not.toContain('stale');
  });

  test('pressing Enter supersedes an instant search still in flight', async () => {
    const { searchMock } = await loadModules(jevCapabilities(true));
    const instant = deferred();
    const manual = deferred();
    searchMock.mockReturnValueOnce(instant.promise).mockReturnValueOnce(manual.promise);

    type('beam loss');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);
    await flush();

    pressEnter();
    await flush();

    manual.resolve(searchResponse(['from-enter']));
    await flush();
    instant.resolve(searchResponse(['from-typing']));
    await flush();

    expect(results().textContent).toContain('from-enter');
    expect(results().textContent).not.toContain('from-typing');
  });

  test('Enter cancels a debounced keystroke instead of running it twice', async () => {
    const { searchMock } = await loadModules(jevCapabilities(true));
    searchMock.mockResolvedValue(searchResponse(['e1']));

    type('beam loss');
    pressEnter();
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);
    await flush();

    expect(searchMock).toHaveBeenCalledTimes(1);
  });

  test('a full search keeps the search button, and gets it back', async () => {
    const { searchMock } = await loadModules(jevCapabilities(true));
    const manual = deferred();
    searchMock.mockReturnValueOnce(manual.promise);
    const button = /** @type {HTMLButtonElement} */ (document.getElementById('search-btn'));

    type('beam loss');
    pressEnter();
    await flush();
    expect(button.disabled).toBe(true);

    // A keystroke arriving mid-search must not bump the sequence the running
    // search checks before handing the button back.
    type('beam losses');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);
    await flush();

    manual.resolve(searchResponse(['e1']));
    await flush();

    expect(button.disabled).toBe(false);
  });
});

describe('failures', () => {
  test('a failed keystroke leaves the previous results alone', async () => {
    const { searchMock } = await loadModules(jevCapabilities(true));
    searchMock.mockResolvedValueOnce(searchResponse(['e1']));

    type('beam loss');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);
    await flush();
    expect(results().textContent).toContain('e1');

    searchMock.mockRejectedValueOnce(new Error('network down'));
    type('beam losses');
    await vi.advanceTimersByTimeAsync(PAST_DEBOUNCE_MS);
    await flush();

    // No error state painted: the operator did not ask for this search, and the
    // next keystroke will try again.
    expect(results().textContent).toContain('e1');
    expect(results().textContent).not.toContain('Search Failed');
  });
});
