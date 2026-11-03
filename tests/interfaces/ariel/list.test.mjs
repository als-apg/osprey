// @ts-check
/**
 * The ARIEL list pane (list.js):
 *   npx vitest run tests/interfaces/ariel/list.test.mjs
 *
 * `initList(root, {capabilities})` builds the pane head (standalone search
 * box, mode chip with its Filters popover, standalone compose button) and
 * follows the store: an empty query pages through /api/entries newest first
 * with day separators and a Load older control; any other query runs the
 * search driver. The mailbox scopes both: Mine passes its author (and locks the
 * Filters author), Drafts shows its empty state. Row click selects. The
 * two-phase paint cases live in list-paint.test.mjs, the banners in
 * vocabulary-banner.test.mjs, the sink-security cases in render-xss.test.mjs.
 *
 * Runs under happy-dom. Only api.js is mocked (the network boundary). The store
 * is a module singleton, so every case resets modules and imports the store and
 * the list from the same reset.
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

/** Two plain modes plus a shared `author` filter, as /api/capabilities sends them. */
const CAPABILITIES = /** @type {any} */ ({
  categories: {
    direct: {
      label: 'Direct',
      modes: [
        { name: 'keyword', label: 'Keyword', parameters: [] },
        {
          name: 'semantic',
          label: 'Semantic',
          parameters: [
            {
              name: 'similarity_threshold',
              label: 'Similarity Threshold',
              type: 'float',
              default: 0.5,
              min: 0,
              max: 1,
              step: 0.1,
              section: 'Retrieval',
            },
          ],
        },
      ],
    },
  },
  default_mode: 'keyword',
  shared_parameters: [
    {
      name: 'author',
      label: 'Author',
      type: 'text',
      default: null,
      section: 'Filters',
    },
  ],
  vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
});

/** @type {any} */
let api;
/** @type {any} */
let store;
/** @type {any} */
let list;

/**
 * @param {string} id
 * @param {string} [timestamp]
 * @returns {any}
 */
function entry(id, timestamp = '2026-08-25T12:00:00Z') {
  return {
    entry_id: id,
    timestamp,
    author: 'alice',
    source_system: 'demo',
    raw_text: `subject of ${id}\nbody of ${id}`,
    score: null,
    attachments: [],
    keywords: [],
    highlights: [],
  };
}

/**
 * @param {any[]} entries
 * @param {{page?: number, total_pages?: number}} [paging]
 * @returns {any}
 */
function listPage(entries, { page = 1, total_pages = 1 } = {}) {
  return { entries, total: entries.length, page, total_pages };
}

/**
 * @param {any[]} entries
 * @returns {any}
 */
function searchResponse(entries) {
  return {
    answer: '',
    sources: [],
    search_modes_used: ['keyword'],
    execution_time_ms: 3,
    total_results: entries.length,
    entries,
    diagnostics: [],
    expanded_terms: [],
  };
}

/** Let every pending microtask (and the awaits chained behind it) run. */
async function flush() {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Mount the list pane from a fresh module graph.
 * @param {{capabilities?: any, state?: any, firstPage?: any}} [opts]
 */
async function mount({ capabilities = CAPABILITIES, state = {}, firstPage = listPage([]) } = {}) {
  api = await import(API_PATH);
  store = await import(STORE_PATH);
  list = await import(LIST_PATH);
  store.setState(state);
  vi.mocked(api.entriesApi.list).mockResolvedValue(firstPage);
  vi.mocked(api.searchApi.search).mockResolvedValue(searchResponse([]));
  list.initList(root(), { capabilities });
  await flush();
}

/** @returns {HTMLElement} */
function root() {
  return /** @type {HTMLElement} */ (document.getElementById('ariel-list'));
}

/** @returns {HTMLElement} */
function body() {
  return /** @type {HTMLElement} */ (document.getElementById('ariel-list-body'));
}

/** @returns {HTMLInputElement} */
function searchInput() {
  return /** @type {HTMLInputElement} */ (document.getElementById('ariel-search-input'));
}

/** @returns {HTMLButtonElement} */
function chip() {
  return /** @type {HTMLButtonElement} */ (document.getElementById('ariel-mode-chip'));
}

/** @returns {HTMLElement} */
function popover() {
  return /** @type {HTMLElement} */ (document.getElementById('ariel-filters-popover'));
}

/**
 * Type into the search box and press Enter.
 * @param {string} value
 */
function enter(value) {
  const input = searchInput();
  input.value = value;
  input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
}

/** @returns {string[]} The day separators' labels, in order. */
function separators() {
  return Array.from(body().querySelectorAll('.ariel-day-sep')).map((el) => el.textContent ?? '');
}

/** @returns {string[]} The painted rows' entry ids, in order. */
function rowIds() {
  return Array.from(body().querySelectorAll('[data-entry-id]')).map(
    (el) => /** @type {HTMLElement} */ (el).dataset.entryId ?? ''
  );
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  localStorage.clear();
  history.replaceState(null, '', '/');
  document.documentElement.setAttribute('data-facility-timezone', 'UTC');
  document.body.innerHTML = '<section id="ariel-list"></section>';
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
  history.replaceState(null, '', '/');
  document.documentElement.removeAttribute('data-facility-timezone');
});

describe('pane head', () => {
  test('standalone: search box, mode chip and compose button, inside a pane head', async () => {
    await mount();
    const head = /** @type {HTMLElement} */ (root().querySelector('.osprey-pane-head'));
    expect(head).not.toBeNull();
    expect(head.querySelector('.osprey-search input')).toBe(searchInput());
    expect(head.contains(chip())).toBe(true);
    expect(chip().textContent, 'chip names the default mode').toContain('Keyword');
    const compose = /** @type {HTMLElement} */ (document.getElementById('ariel-compose-btn'));
    expect(head.contains(compose)).toBe(true);
    expect(compose.getAttribute('aria-label')).toBe('New entry');
  });

  test('embedded: only the mode chip — the search box and compose belong to the host', async () => {
    history.replaceState(null, '', '/?embedded=true');
    await mount();
    expect(searchInput()).toBeNull();
    expect(document.getElementById('ariel-compose-btn')).toBeNull();
    expect(chip()).not.toBeNull();
  });

  test('the chip follows the store mode', async () => {
    await mount();
    store.setState({ mode: 'semantic', rerank: 'now' });
    expect(chip().textContent).toContain('Semantic');
  });

  test('the compose button opens an empty compose', async () => {
    await mount();
    /** @type {HTMLElement} */ (document.getElementById('ariel-compose-btn')).click();
    expect(store.getState().compose).toEqual({ draft: null });
  });

  test('a query written elsewhere shows in the search box', async () => {
    await mount();
    store.setState({ query: 'from the hash', rerank: 'now' });
    expect(searchInput().value).toBe('from the hash');
  });
});

describe('search box', () => {
  test('Enter writes the query with rerank now and runs the search', async () => {
    await mount();
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(searchResponse([entry('HIT-1')]));
    enter('  quench  ');
    expect(store.getState().query).toBe('quench');
    expect(store.getState().rerank).toBe('now');
    await flush();
    expect(api.searchApi.search).toHaveBeenCalledTimes(1);
    expect(vi.mocked(api.searchApi.search).mock.calls[0][0]).toMatchObject({
      query: 'quench',
      mode: 'keyword',
    });
    expect(rowIds()).toEqual(['HIT-1']);
  });

  test('Enter on an unchanged query re-runs it (forced write)', async () => {
    await mount();
    enter('quench');
    await flush();
    enter('quench');
    await flush();
    expect(api.searchApi.search).toHaveBeenCalledTimes(2);
  });

  test('a search shows the single result set: no day separators, no Load older', async () => {
    await mount();
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([entry('A', '2026-08-25T12:00:00Z'), entry('B', '2026-08-20T12:00:00Z')])
    );
    enter('quench');
    await flush();
    expect(rowIds()).toEqual(['A', 'B']);
    expect(separators()).toEqual([]);
    expect(body().querySelector('[data-load-older]')).toBeNull();
  });

  test('focusSearch focuses the standalone box', async () => {
    await mount();
    list.focusSearch();
    expect(document.activeElement).toBe(searchInput());
  });

  test('focusSearch is a no-op embedded and while search is blocked', async () => {
    history.replaceState(null, '', '/?embedded=true');
    await mount();
    expect(() => list.focusSearch()).not.toThrow();

    vi.resetModules();
    history.replaceState(null, '', '/');
    document.body.innerHTML = '<section id="ariel-list"></section>';
    await mount({
      capabilities: { ...CAPABILITIES, status: 'configuration_invalid', config_errors: ['x'] },
    });
    list.focusSearch();
    expect(document.activeElement).not.toBe(searchInput());
  });

  test('clearing the query returns the list to newest-first', async () => {
    await mount();
    enter('quench');
    await flush();
    vi.mocked(api.entriesApi.list).mockClear();
    vi.mocked(api.entriesApi.list).mockResolvedValueOnce(listPage([entry('NEWEST')]));
    enter('');
    await flush();
    expect(api.entriesApi.list).toHaveBeenCalledTimes(1);
    expect(rowIds()).toEqual(['NEWEST']);
  });
});

describe('listing (empty query)', () => {
  test('lists /api/entries page 1, newest first, with a separator per day', async () => {
    await mount({
      firstPage: listPage([
        entry('E3', '2026-08-25T15:00:00Z'),
        entry('E2', '2026-08-25T09:00:00Z'),
        entry('E1', '2026-08-24T22:00:00Z'),
      ]),
    });
    expect(vi.mocked(api.entriesApi.list).mock.calls[0][0]).toMatchObject({ page: 1 });
    expect(rowIds()).toEqual(['E3', 'E2', 'E1']);
    const days = separators();
    expect(days.length, 'one separator per calendar day').toBe(2);
    expect(days[0]).toContain('25');
    expect(days[1]).toContain('24');
    // Each separator precedes its day's rows.
    const order = Array.from(body().querySelectorAll('.ariel-day-sep, [data-entry-id]')).map(
      (el) =>
        el.classList.contains('ariel-day-sep')
          ? 'day'
          : /** @type {HTMLElement} */ (el).dataset.entryId
    );
    expect(order).toEqual(['day', 'E3', 'E2', 'day', 'E1']);
  });

  test('day separators follow the facility time zone', async () => {
    document.documentElement.setAttribute('data-facility-timezone', 'America/Los_Angeles');
    // 02:00 UTC on the 25th is still the 24th in Berkeley.
    await mount({
      firstPage: listPage([
        entry('LATE', '2026-08-25T02:00:00Z'),
        entry('EARLIER', '2026-08-24T20:00:00Z'),
      ]),
    });
    expect(separators().length, 'both on one facility day').toBe(1);
    expect(separators()[0]).toContain('24');
  });

  test('Load older appends the next page and continues the day without a repeat separator', async () => {
    await mount({
      firstPage: listPage([entry('P1', '2026-08-25T12:00:00Z')], { page: 1, total_pages: 2 }),
    });
    expect(separators().length).toBe(1);
    vi.mocked(api.entriesApi.list).mockResolvedValueOnce(
      listPage(
        [entry('P2-SAME-DAY', '2026-08-25T08:00:00Z'), entry('P2-OLDER', '2026-08-23T08:00:00Z')],
        { page: 2, total_pages: 2 }
      )
    );
    /** @type {HTMLElement} */ (body().querySelector('[data-load-older]')).click();
    await flush();
    expect(vi.mocked(api.entriesApi.list).mock.calls.at(-1)?.[0]).toMatchObject({ page: 2 });
    expect(rowIds()).toEqual(['P1', 'P2-SAME-DAY', 'P2-OLDER']);
    expect(separators().length, 'only the new day adds a separator').toBe(2);
    expect(body().querySelector('[data-load-older]'), 'no more pages').toBeNull();
  });

  test('a failed listing shows the error state', async () => {
    api = await import(API_PATH);
    store = await import(STORE_PATH);
    list = await import(LIST_PATH);
    vi.mocked(api.entriesApi.list).mockRejectedValueOnce(new Error('HTTP 503: db down'));
    list.initList(root(), { capabilities: CAPABILITIES });
    await flush();
    expect(body().querySelector('.text-error')?.textContent).toContain('Failed to Load Entries');
    expect(body().getAttribute('aria-busy')).toBe('false');
  });

  test('a query while search is blocked shows the listing, not a search', async () => {
    await mount({
      capabilities: { ...CAPABILITIES, status: 'configuration_invalid', config_errors: ['x'] },
      state: { query: 'from the hash' },
      firstPage: listPage([entry('NEWEST')]),
    });
    expect(api.searchApi.search).not.toHaveBeenCalled();
    expect(rowIds()).toEqual(['NEWEST']);
  });

  test('an empty listing shows the empty state', async () => {
    await mount();
    expect(body().textContent).toContain('No Entries');
  });

  test('a mode or filters change does not reload the newest-first listing', async () => {
    await mount();
    vi.mocked(api.entriesApi.list).mockClear();
    store.setState({ mode: 'semantic', rerank: 'now' });
    store.setState({ filters: { similarity_threshold: 0.8 }, rerank: 'now' });
    await flush();
    expect(api.entriesApi.list).not.toHaveBeenCalled();
    expect(api.searchApi.search).not.toHaveBeenCalled();
  });
});

describe('mailboxes', () => {
  test('Mine passes its author to the listing and to the search', async () => {
    await mount({ state: { mailbox: 'mine', mineAuthor: 'thorsten' } });
    expect(vi.mocked(api.entriesApi.list).mock.calls[0][0]).toMatchObject({ author: 'thorsten' });

    enter('quench');
    await flush();
    expect(vi.mocked(api.searchApi.search).mock.calls[0][0].advancedParams).toMatchObject({
      author: 'thorsten',
    });
  });

  test('Mine locks the Filters author to the Mine name', async () => {
    await mount({ state: { mailbox: 'mine', mineAuthor: 'thorsten' } });
    chip().click();
    const author = /** @type {HTMLInputElement} */ (document.getElementById('param-author'));
    expect(author, 'author control rendered').not.toBeNull();
    expect(author.disabled).toBe(true);
    expect(author.value).toBe('thorsten');
  });

  test('outside Mine the Filters author is free', async () => {
    await mount();
    chip().click();
    const author = /** @type {HTMLInputElement} */ (document.getElementById('param-author'));
    expect(author.disabled).toBe(false);
  });

  test('Mine without a known name lists nothing and says why', async () => {
    await mount({ state: { mailbox: 'mine', mineAuthor: null } });
    expect(api.entriesApi.list).not.toHaveBeenCalled();
    expect(body().textContent).toContain('Mine Needs a Name');

    store.setState({ mineAuthor: 'thorsten' });
    await flush();
    expect(vi.mocked(api.entriesApi.list).mock.calls[0][0]).toMatchObject({ author: 'thorsten' });
  });

  test('Drafts shows its empty state, with or without a query, and requests nothing', async () => {
    await mount({ state: { mailbox: 'drafts' } });
    expect(body().textContent).toContain(list.DRAFTS_EMPTY.text);
    enter('quench');
    await flush();
    expect(body().textContent).toContain(list.DRAFTS_EMPTY.text);
    expect(api.entriesApi.list).not.toHaveBeenCalled();
    expect(api.searchApi.search).not.toHaveBeenCalled();
  });

  test('switching mailbox re-runs the list', async () => {
    await mount();
    vi.mocked(api.entriesApi.list).mockClear();
    store.setState({ mailbox: 'mine', mineAuthor: 'thorsten', rerank: 'now' });
    await flush();
    expect(vi.mocked(api.entriesApi.list).mock.calls[0][0]).toMatchObject({ author: 'thorsten' });
    store.setState({ mailbox: 'all', rerank: 'now' });
    await flush();
    expect(vi.mocked(api.entriesApi.list).mock.calls[1][0].author).toBeUndefined();
  });
});

describe('selection', () => {
  test('a row click selects the entry and shows the reader', async () => {
    await mount({ firstPage: listPage([entry('E1'), entry('E2')]) });
    /** @type {HTMLElement} */ (body().querySelector('[data-entry-id="E2"]')).click();
    expect(store.getState().sel).toBe('E2');
    expect(store.getState().narrowView).toBe('reader');
  });

  test('rows are list-row primitives and mirror the selection', async () => {
    await mount({ firstPage: listPage([entry('E1'), entry('E2')]) });
    const rows = /** @type {HTMLElement[]} */ (
      Array.from(body().querySelectorAll('[data-entry-id]'))
    );
    expect(rows.every((r) => r.classList.contains('osprey-list-row'))).toBe(true);
    expect(rows.map((r) => r.getAttribute('aria-selected'))).toEqual(['false', 'false']);
    store.setState({ sel: 'E1' });
    expect(rows.map((r) => r.getAttribute('aria-selected'))).toEqual(['true', 'false']);
  });

  test('a row shows subject, author, time, score, preview, attachments and keywords', async () => {
    await mount({
      firstPage: listPage([
        {
          ...entry('E1'),
          score: 0.91,
          attachments: [{ filename: 'a.png' }],
          keywords: ['rf', 'quench'],
        },
      ]),
    });
    const row = /** @type {HTMLElement} */ (body().querySelector('[data-entry-id="E1"]'));
    expect(row.querySelector('.ariel-row-subject')?.textContent).toBe('subject of E1');
    expect(row.textContent).toContain('alice');
    expect(row.querySelector('time')?.getAttribute('datetime')).toBe('2026-08-25T12:00:00Z');
    expect(row.querySelector('.score-badge')?.textContent).toBe('91%');
    expect(row.querySelector('.ariel-row-preview')?.textContent).toContain('body of E1');
    expect(row.querySelector('[data-attachment-count]')?.textContent).toContain('1');
    expect(Array.from(row.querySelectorAll('.keyword-tag')).map((k) => k.textContent)).toEqual([
      'rf',
      'quench',
    ]);
  });

  test('a row shows sanitized highlights instead of the preview when there are some', async () => {
    await mount({
      firstPage: listPage([{ ...entry('E1'), highlights: ['the <b>quench</b> <i>tripped</i>'] }]),
    });
    const preview = /** @type {HTMLElement} */ (body().querySelector('.ariel-row-preview'));
    expect(preview.querySelector('b')?.textContent).toBe('quench');
    expect(preview.querySelector('i'), 'only <b> survives').toBeNull();
    expect(preview.textContent).not.toContain('body of E1');
  });
});

describe('mode chip and Filters popover', () => {
  test('the chip opens a body-level popover with the mode choices and filters', async () => {
    await mount();
    expect(popover().hidden).toBe(true);
    expect(popover().parentElement).toBe(document.body);
    chip().click();
    expect(popover().hidden).toBe(false);
    expect(popover().querySelectorAll('[data-mode]').length).toBe(2);
    expect(
      /** @type {HTMLElement} */ (popover().querySelector('[data-mode="keyword"]')).getAttribute(
        'aria-pressed'
      )
    ).toBe('true');
  });

  test('picking a mode writes mode with rerank now and re-runs the query', async () => {
    await mount({ state: { query: 'quench' } });
    vi.mocked(api.searchApi.search).mockClear();
    chip().click();
    /** @type {HTMLElement} */ (popover().querySelector('[data-mode="semantic"]')).click();
    expect(store.getState().mode).toBe('semantic');
    expect(store.getState().rerank).toBe('now');
    await flush();
    expect(vi.mocked(api.searchApi.search).mock.calls[0][0].mode).toBe('semantic');
    // The popover re-rendered for the new mode's parameters.
    expect(document.getElementById('param-similarity_threshold')).not.toBeNull();
  });

  test('a filters change writes filters with rerank now and re-runs the query', async () => {
    await mount({ state: { query: 'quench' } });
    vi.mocked(api.searchApi.search).mockClear();
    chip().click();
    const author = /** @type {HTMLInputElement} */ (document.getElementById('param-author'));
    author.value = 'bob';
    author.dispatchEvent(new Event('input'));
    expect(store.getState().filters).toEqual({ author: 'bob' });
    expect(store.getState().rerank).toBe('now');
    expect(document.getElementById('param-author'), 'the field being typed in survives').toBe(
      author
    );
    await flush();
    expect(vi.mocked(api.searchApi.search).mock.calls[0][0].advancedParams).toMatchObject({
      author: 'bob',
    });
  });
});

describe('rerank pass-through', () => {
  test("a when-stable query is not reranked until the driver's settle window passes", async () => {
    const hybrid = /** @type {any} */ ({
      ...CAPABILITIES,
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
    });
    await mount({ capabilities: hybrid });
    vi.useFakeTimers();
    try {
      store.setState({ query: 'typed in the tile bar', rerank: 'when-stable' });
      await vi.advanceTimersByTimeAsync(0);
      expect(api.searchApi.search, 'phase 1 only').toHaveBeenCalledTimes(1);
      await vi.advanceTimersByTimeAsync(1500);
      expect(api.searchApi.search, 'phase 2 after the window').toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });
});
