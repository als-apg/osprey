// @ts-check
/**
 * The ARIEL list pane (list.js):
 *   npx vitest run tests/interfaces/ariel/list.test.mjs
 *
 * `initList(root, {capabilities})` builds the pane head (standalone search
 * box, mode chip with its Filters popover, standalone compose button) and
 * follows the store: an empty query browses /api/entries newest first by
 * cursor, with day separators and a Load older control; any other query runs
 * the search driver and pages by offset, in ranked order. Both apply the
 * filter; the mailbox scopes both: Mine sends its name as the exact author
 * (and locks the Filters author), Drafts hands the body to drafts-list.js
 * (its own cases live in drafts-list.test.mjs). The store's
 * notice shows in the banners. Row click selects. paging.js's own cases live
 * in paging.test.mjs. The
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
    draftsApi: { ...actual.draftsApi, list: vi.fn() },
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
    subject: `subject of ${id}`,
    snippet: `body of ${id}`,
    score: null,
    attachment_count: 0,
    has_images: false,
    enhancements: {},
    keywords: [],
    highlights: [],
    matched_via: [],
  };
}

/**
 * @param {any[]} entries
 * @param {{next_cursor?: string|null, total?: number}} [paging]
 * @returns {any}
 */
function listPage(entries, { next_cursor = null, total = entries.length } = {}) {
  return { entries, total, page: null, page_size: 20, total_pages: 1, next_cursor };
}

/**
 * @param {any[]} entries
 * @param {{total?: number, total_exact?: boolean, has_more?: boolean}} [paging]
 * @returns {any}
 */
function searchResponse(
  entries,
  { total = entries.length, total_exact = true, has_more = false } = {}
) {
  return {
    answer: '',
    sources: [],
    search_modes_used: ['keyword'],
    execution_time_ms: 3,
    total_results: total,
    total,
    total_exact,
    has_more,
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
  vi.mocked(api.draftsApi.list).mockResolvedValue({ drafts: [] });
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

/** @returns {HTMLButtonElement|null} The Load more control. */
function loadMoreButton() {
  return body().querySelector('[data-load-more]');
}

/** @returns {string} The header's total. */
function totalText() {
  return body().querySelector('[data-list-total]')?.textContent ?? '';
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

  test('a search shows ranked rows without day separators, and no Load more at the end', async () => {
    await mount();
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([entry('A', '2026-08-25T12:00:00Z'), entry('B', '2026-08-20T12:00:00Z')])
    );
    enter('quench');
    await flush();
    expect(vi.mocked(api.searchApi.search).mock.calls[0][0]).toMatchObject({
      offset: 0,
      maxResults: list.PAGE_SIZE,
      view: 'row',
    });
    expect(rowIds()).toEqual(['A', 'B']);
    expect(separators()).toEqual([]);
    expect(loadMoreButton(), 'has_more false: the end').toBeNull();
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
  test('lists the first /api/entries page, newest first, with a separator per day', async () => {
    await mount({
      firstPage: listPage([
        entry('E3', '2026-08-25T15:00:00Z'),
        entry('E2', '2026-08-25T09:00:00Z'),
        entry('E1', '2026-08-24T22:00:00Z'),
      ]),
    });
    const sent = vi.mocked(api.entriesApi.list).mock.calls[0][0];
    expect(sent).toMatchObject({ pageSize: list.PAGE_SIZE, view: 'row' });
    expect(sent.page, 'no page numbers').toBeUndefined();
    expect(sent.cursor, 'the first page has no cursor').toBeUndefined();
    expect(rowIds()).toEqual(['E3', 'E2', 'E1']);
    expect(totalText(), 'the header shows the total').toBe('3');
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

  test('Load older continues from next_cursor and continues the day without a repeat separator', async () => {
    await mount({
      firstPage: listPage([entry('P1', '2026-08-25T12:00:00Z')], { next_cursor: 'C1', total: 3 }),
    });
    expect(separators().length).toBe(1);
    expect(loadMoreButton()?.textContent).toContain('Load older');
    vi.mocked(api.entriesApi.list).mockResolvedValueOnce(
      listPage(
        [entry('P2-SAME-DAY', '2026-08-25T08:00:00Z'), entry('P2-OLDER', '2026-08-23T08:00:00Z')],
        { next_cursor: null, total: 3 }
      )
    );
    /** @type {HTMLElement} */ (loadMoreButton()).click();
    await flush();
    expect(vi.mocked(api.entriesApi.list).mock.calls.at(-1)?.[0]).toMatchObject({
      cursor: 'C1',
      view: 'row',
    });
    expect(rowIds()).toEqual(['P1', 'P2-SAME-DAY', 'P2-OLDER']);
    expect(separators().length, 'only the new day adds a separator').toBe(2);
    expect(loadMoreButton(), 'no cursor: the end').toBeNull();
    expect(body().getAttribute('aria-busy')).toBe('false');
  });

  test('a failed Load older keeps the rows, says why, and can be retried', async () => {
    await mount({ firstPage: listPage([entry('P1')], { next_cursor: 'C1' }) });
    vi.mocked(api.entriesApi.list).mockRejectedValueOnce(new Error('HTTP 503: db down'));
    /** @type {HTMLElement} */ (loadMoreButton()).click();
    await flush();
    expect(rowIds()).toEqual(['P1']);
    expect(body().querySelector('[data-load-more-error]')?.textContent).toContain('db down');
    expect(loadMoreButton()?.disabled).toBe(false);
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

  test('a filter change re-runs the listing with the filter and starts paging over', async () => {
    await mount({ firstPage: listPage([entry('P1')], { next_cursor: 'C1' }) });
    vi.mocked(api.entriesApi.list).mockClear();
    store.setState({ filter: { source_system: ['ops'] } });
    await flush();
    const sent = vi.mocked(api.entriesApi.list).mock.calls[0][0];
    expect(sent.filters).toEqual({ source_system: ['ops'] });
    expect(sent.cursor, 'paging starts over').toBeUndefined();
    expect(api.searchApi.search).not.toHaveBeenCalled();
  });

  test('a Filters author knob reaches the listing as a filter dimension', async () => {
    await mount();
    vi.mocked(api.entriesApi.list).mockClear();
    store.setState({ filters: { author: 'bob' }, rerank: 'now' });
    await flush();
    expect(vi.mocked(api.entriesApi.list).mock.calls[0][0].filters).toEqual({ author: ['bob'] });
  });
});

describe('mailboxes', () => {
  test('Mine sends its name as the exact author filter to the listing and to the search', async () => {
    await mount({
      state: { mailbox: 'mine', mineAuthor: 'thorsten', filter: { author: ['alice'] } },
    });
    const listed = vi.mocked(api.entriesApi.list).mock.calls[0][0];
    expect(listed.filters).toEqual({ author: ['thorsten'] });
    expect(listed.author, 'no legacy author param').toBeUndefined();

    enter('quench');
    await flush();
    const searched = vi.mocked(api.searchApi.search).mock.calls[0][0];
    expect(searched.filters).toEqual({ author: ['thorsten'] });
    expect(searched.advancedParams.author).toBeUndefined();
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
    expect(vi.mocked(api.entriesApi.list).mock.calls[0][0].filters).toEqual({
      author: ['thorsten'],
    });
  });

  test('Drafts lists the drafts, with or without a query, and never entries or search', async () => {
    await mount({ state: { mailbox: 'drafts' } });
    expect(api.draftsApi.list).toHaveBeenCalledTimes(1);
    expect(body().textContent).toContain('No Drafts');
    enter('quench');
    await flush();
    expect(api.draftsApi.list).toHaveBeenCalledTimes(2);
    expect(body().textContent).toContain('No Drafts');
    expect(api.entriesApi.list).not.toHaveBeenCalled();
    expect(api.searchApi.search).not.toHaveBeenCalled();
    expect(loadMoreButton(), 'Drafts has no Load more').toBeNull();
    expect(body().getAttribute('aria-busy')).toBe('false');
  });

  test('leaving Drafts before the drafts answer keeps the listing painted', async () => {
    await mount({ firstPage: listPage([entry('e1')]) });
    /** @type {(value: any) => void} */
    let answer = () => {};
    vi.mocked(api.draftsApi.list).mockReturnValue(new Promise((resolve) => (answer = resolve)));
    store.setState({ mailbox: 'drafts' });
    store.setState({ mailbox: 'all' });
    await flush();
    answer({ drafts: [{ draft_id: 'd1', subject: 'late', created_at: '2026-08-25T12:00:00Z' }] });
    await flush();
    expect(body().querySelector('[data-draft-id]')).toBeNull();
    expect(body().querySelector('[data-entry-id="e1"]')).not.toBeNull();
  });

  test('switching mailbox re-runs the list', async () => {
    await mount();
    vi.mocked(api.entriesApi.list).mockClear();
    store.setState({ mailbox: 'mine', mineAuthor: 'thorsten', rerank: 'now' });
    await flush();
    expect(vi.mocked(api.entriesApi.list).mock.calls[0][0].filters).toEqual({
      author: ['thorsten'],
    });
    store.setState({ mailbox: 'all', rerank: 'now' });
    await flush();
    expect(vi.mocked(api.entriesApi.list).mock.calls[1][0].filters).toEqual({});
  });
});

describe('selection', () => {
  test('a row click selects the entry and shows the reader', async () => {
    await mount({ firstPage: listPage([entry('E1'), entry('E2')]) });
    /** @type {HTMLElement} */ (body().querySelector('[data-entry-id="E2"]')).click();
    expect(store.getState().sel).toBe('E2');
    expect(store.getState().narrowView).toBe('reader');
  });

  test('a row click while an attachment is open clears att from state and hash', async () => {
    await mount({
      firstPage: listPage([entry('E1'), entry('E2')]),
      state: { sel: 'E1', att: 'a1' },
    });
    const router = await import('../../../src/osprey/interfaces/ariel/static/js/router.js');
    /** @type {HTMLElement} */ (body().querySelector('[data-entry-id="E2"]')).click();
    expect(store.getState()).toMatchObject({ sel: 'E2', att: null });
    expect(router.serializeState(store.getState())).not.toContain('att=');
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
          attachment_count: 1,
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

  test('a row without a subject falls back to the first line of its snippet', async () => {
    await mount({
      firstPage: listPage([{ ...entry('E1'), subject: null, snippet: 'first line\nsecond' }]),
    });
    expect(body().querySelector('.ariel-row-subject')?.textContent).toBe('first line');
    expect(body().querySelector('.ariel-row-preview')?.textContent).toContain('second');
  });
});

/** Two enhancement modules, in registry order. */
const MODULES = [
  { name: 'text_embedding', label: 'Text embedding' },
  { name: 'image_caption', label: 'Image caption' },
];

describe('badge strip', () => {
  /** @param {any} enhancements @param {any[]} modules */
  async function strip(enhancements, modules = MODULES) {
    const components = await import('../../../src/osprey/interfaces/ariel/static/js/components.js');
    const host = document.createElement('div');
    host.innerHTML = components.renderBadgeStrip(enhancements, modules);
    return host;
  }

  test.each([
    ['complete', 'complete'],
    ['pending', 'pending'],
    ['stale', 'stale'],
    ['failed', 'failed'],
  ])('a %s cell carries its state, class and text alternative', async (state, token) => {
    const host = await strip({ image_caption: state });
    const cell = /** @type {HTMLElement} */ (host.querySelector('[data-module="image_caption"]'));
    expect(cell.dataset.state).toBe(state);
    expect(cell.classList.contains(`ariel-strip-cell-${token}`)).toBe(true);
    expect(cell.getAttribute('role')).toBe('img');
    expect(cell.getAttribute('aria-label')).toBe(`Image caption: ${state}`);
    expect(cell.getAttribute('title')).toBe(`Image caption: ${state}`);
  });

  test('a module that never ran reads "not run", not "pending"', async () => {
    const host = await strip({ image_caption: 'not_run', text_embedding: 'pending' });
    const never = /** @type {HTMLElement} */ (host.querySelector('[data-module="image_caption"]'));
    expect(never.dataset.state).toBe('not_run');
    expect(never.classList.contains('ariel-strip-cell-not-run')).toBe(true);
    expect(never.getAttribute('aria-label')).toBe('Image caption: not run');
    const queued = /** @type {HTMLElement} */ (
      host.querySelector('[data-module="text_embedding"]')
    );
    expect(queued.getAttribute('aria-label')).toBe('Text embedding: pending');
  });

  test('one cell per module that applies, in module order; n/a cells render nothing', async () => {
    const host = await strip({ text_embedding: 'pending', image_caption: 'n/a' });
    const cells = Array.from(host.querySelectorAll('.ariel-strip-cell'));
    expect(cells.map((c) => /** @type {HTMLElement} */ (c).dataset.module)).toEqual([
      'text_embedding',
    ]);
    expect(host.querySelector('.ariel-strip')?.getAttribute('aria-label')).toBe('Enhancements');
  });

  test('missing, unknown or n/a states everywhere: no strip at all', async () => {
    expect((await strip({ text_embedding: 'bogus', other_module: 'complete' })).innerHTML).toBe('');
    expect((await strip({ text_embedding: 'n/a', image_caption: 'n/a' })).innerHTML).toBe('');
    expect((await strip(undefined)).innerHTML).toBe('');
  });

  test('no modules: no strip', async () => {
    expect((await strip({ a: 'complete' }, [])).innerHTML).toBe('');
    expect((await strip(undefined, /** @type {any} */ (null))).innerHTML).toBe('');
  });

  test('renderListRow renders the strip from enhancements only when strip is on', async () => {
    const components = await import('../../../src/osprey/interfaces/ariel/static/js/components.js');
    components.setEnhancementModules(MODULES);
    const row = { ...entry('E1'), enhancements: { text_embedding: 'complete' } };
    const on = document.createElement('div');
    on.innerHTML = components.renderListRow(row, null, { strip: true });
    expect(on.querySelector('[data-module="text_embedding"]')?.getAttribute('data-state')).toBe(
      'complete'
    );
    const off = document.createElement('div');
    off.innerHTML = components.renderListRow(row, null, { strip: false });
    expect(off.querySelector('.ariel-strip')).toBeNull();
  });

  test('the list follows the preference: rows repaint when it changes', async () => {
    const components = await import('../../../src/osprey/interfaces/ariel/static/js/components.js');
    components.setEnhancementModules(MODULES);
    await mount({
      firstPage: listPage([{ ...entry('E1'), enhancements: { image_caption: 'stale' } }]),
    });
    expect(
      body().querySelector('.ariel-strip-cell-stale'),
      'standalone default: on'
    ).not.toBeNull();

    const prefs = await import('../../../src/osprey/interfaces/ariel/static/js/display-prefs.js');
    prefs.setStripPref(false);
    await flush();
    expect(body().querySelector('[data-entry-id="E1"]')).not.toBeNull();
    expect(body().querySelector('.ariel-strip')).toBeNull();

    prefs.setStripPref(true);
    await flush();
    expect(body().querySelector('.ariel-strip-cell-stale')).not.toBeNull();
  });

  test('re-initialising the list does not stack preference subscriptions', async () => {
    await mount({ firstPage: listPage([entry('E1')]) });
    list.initList(root(), { capabilities: CAPABILITIES });
    await flush();
    const calls = vi.mocked(api.entriesApi.list).mock.calls.length;
    const prefs = await import('../../../src/osprey/interfaces/ariel/static/js/display-prefs.js');
    prefs.setStripPref(false);
    await flush();
    expect(vi.mocked(api.entriesApi.list).mock.calls.length).toBe(calls + 1);
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

  test('dragging a filter slider issues at most one search, after the drag settles', async () => {
    await mount({ state: { query: 'quench', mode: 'semantic' } });
    vi.mocked(api.searchApi.search).mockClear();
    chip().click();
    const slider = /** @type {HTMLInputElement} */ (
      document.getElementById('param-similarity_threshold')
    );
    expect(slider.type).toBe('range');
    for (const v of ['0.5', '0.6', '0.7', '0.8']) {
      slider.value = v;
      slider.dispatchEvent(new Event('input'));
      await flush();
    }
    expect(api.searchApi.search, 'no request while dragging').not.toHaveBeenCalled();
    slider.dispatchEvent(new Event('change'));
    await flush();
    expect(api.searchApi.search).toHaveBeenCalledTimes(1);
    expect(
      vi.mocked(api.searchApi.search).mock.calls[0][0].advancedParams.similarity_threshold
    ).toBe(0.8);
  });

  test('a Filters author writes filter.author with rerank now and re-runs the query', async () => {
    await mount({ state: { query: 'quench' } });
    vi.mocked(api.searchApi.search).mockClear();
    chip().click();
    const author = /** @type {HTMLInputElement} */ (document.getElementById('param-author'));
    author.value = 'bob';
    author.dispatchEvent(new Event('input'));
    author.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(store.getState().filter).toEqual({ author: ['bob'] });
    expect(store.getState().filters, 'the author is a dimension, not a knob').toEqual({});
    expect(store.getState().rerank).toBe('now');
    expect(document.getElementById('param-author'), 'the field being typed in survives').toBe(
      author
    );
    await flush();
    expect(vi.mocked(api.searchApi.search).mock.calls[0][0].filters).toEqual({ author: ['bob'] });
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

describe('search paging', () => {
  test('Load more requests the next offset unreranked and appends only ids not shown', async () => {
    await mount();
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([entry('A'), entry('B')], { total: 4, has_more: true })
    );
    enter('quench');
    await flush();
    expect(loadMoreButton()?.textContent).toContain('Load more');
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([entry('B'), entry('C')], { total: 4, has_more: false })
    );
    /** @type {HTMLElement} */ (loadMoreButton()).click();
    await flush();
    expect(api.searchApi.search).toHaveBeenCalledTimes(2);
    expect(vi.mocked(api.searchApi.search).mock.calls[1][0]).toMatchObject({
      query: 'quench',
      offset: 2,
      maxResults: list.PAGE_SIZE,
      view: 'row',
    });
    expect(rowIds(), 'B is not repeated').toEqual(['A', 'B', 'C']);
    expect(separators(), 'ranked order: no day separators').toEqual([]);
    expect(loadMoreButton(), 'has_more false: the end').toBeNull();
  });

  test('an inexact total shows as a lower bound while more pages remain, else as an estimate', async () => {
    await mount();
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([entry('A')], { total: 20, total_exact: false, has_more: true })
    );
    enter('quench');
    await flush();
    expect(totalText()).toBe('20+');
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([entry('B')], { total: 30, total_exact: false, has_more: false })
    );
    /** @type {HTMLElement} */ (loadMoreButton()).click();
    await flush();
    expect(totalText()).toBe('≈30');
  });

  test('an exact total shows as is', async () => {
    await mount();
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([entry('A')], { total: 57, has_more: true })
    );
    enter('quench');
    await flush();
    expect(totalText()).toBe('57');
  });

  test('a new query starts paging over from offset 0', async () => {
    await mount();
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([entry('A')], { total: 2, has_more: true })
    );
    enter('quench');
    await flush();
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(searchResponse([entry('Z')]));
    enter('vacuum');
    await flush();
    expect(vi.mocked(api.searchApi.search).mock.calls[1][0]).toMatchObject({
      query: 'vacuum',
      offset: 0,
    });
    expect(rowIds()).toEqual(['Z']);
    expect(loadMoreButton()).toBeNull();
  });

  test('changing the query after Load more starts again from page 1', async () => {
    await mount();
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([entry('A'), entry('B')], { total: 6, has_more: true })
    );
    enter('quench');
    await flush();
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([entry('C'), entry('D')], { total: 6, has_more: true })
    );
    /** @type {HTMLElement} */ (loadMoreButton()).click();
    await flush();
    expect(vi.mocked(api.searchApi.search).mock.calls[1][0]).toMatchObject({ offset: 2 });
    expect(rowIds()).toEqual(['A', 'B', 'C', 'D']);
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(
      searchResponse([entry('Z')], { total: 1, has_more: false })
    );
    enter('vacuum');
    await flush();
    expect(api.searchApi.search).toHaveBeenCalledTimes(3);
    expect(vi.mocked(api.searchApi.search).mock.calls[2][0]).toMatchObject({
      query: 'vacuum',
      offset: 0,
    });
    expect(rowIds(), 'the earlier pages are gone').toEqual(['Z']);
    expect(loadMoreButton()).toBeNull();
  });

  test('Load more is disabled while the phase-2 rerank is pending', async () => {
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
    /** @type {(value: any) => void} */
    let finishRerank = () => {};
    vi.mocked(api.searchApi.search)
      .mockResolvedValueOnce(searchResponse([entry('FAST')], { total: 5, has_more: true }))
      .mockImplementationOnce(() => new Promise((resolve) => (finishRerank = resolve)));
    enter('quench');
    await flush();
    expect(rowIds()).toEqual(['FAST']);
    expect(loadMoreButton()?.disabled, 'held while reranking').toBe(true);
    /** @type {HTMLElement} */ (loadMoreButton()).click();
    await flush();
    expect(api.searchApi.search, 'no page request while held').toHaveBeenCalledTimes(2);

    finishRerank(searchResponse([entry('RERANKED')], { total: 5, has_more: true }));
    await flush();
    expect(rowIds()).toEqual(['RERANKED']);
    expect(loadMoreButton()?.disabled).toBe(false);
    vi.mocked(api.searchApi.search).mockResolvedValueOnce(searchResponse([entry('NEXT')]));
    /** @type {HTMLElement} */ (loadMoreButton()).click();
    await flush();
    expect(vi.mocked(api.searchApi.search).mock.calls[2][0]).toMatchObject({ offset: 1 });
    expect(vi.mocked(api.searchApi.search).mock.calls[2][0].advancedParams.rerank).toBe(false);
  });
});

describe('superseded requests', () => {
  test('a listing superseded by a search is aborted and never painted', async () => {
    api = await import(API_PATH);
    store = await import(STORE_PATH);
    list = await import(LIST_PATH);
    /** @type {(value: any) => void} */
    let finishListing = () => {};
    vi.mocked(api.entriesApi.list).mockImplementationOnce(
      () => new Promise((resolve) => (finishListing = resolve))
    );
    vi.mocked(api.searchApi.search).mockResolvedValue(searchResponse([entry('HIT')]));
    list.initList(root(), { capabilities: CAPABILITIES });
    await flush();
    const signal = vi.mocked(api.entriesApi.list).mock.calls[0][0].signal;
    enter('quench');
    await flush();
    expect(signal.aborted, 'the listing request was aborted').toBe(true);
    finishListing(listPage([entry('STALE')]));
    await flush();
    expect(rowIds()).toEqual(['HIT']);
    expect(body().getAttribute('aria-busy')).toBe('false');
  });

  test('an aborted search shows no error state', async () => {
    await mount();
    vi.mocked(api.searchApi.search).mockImplementationOnce(
      (/** @type {any} */ params) =>
        new Promise((_resolve, reject) => {
          params.signal.addEventListener('abort', () =>
            reject(new DOMException('aborted', 'AbortError'))
          );
        })
    );
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage([entry('NEWEST')]));
    enter('quench');
    await flush();
    enter('');
    await flush();
    expect(rowIds()).toEqual(['NEWEST']);
    expect(body().querySelector('.text-error')).toBeNull();
  });
});

describe('notice', () => {
  test('a notice in state shows in the banners, after the config banner', async () => {
    await mount({
      capabilities: { ...CAPABILITIES, status: 'configuration_warning', config_errors: ['x'] },
      state: { notice: 'Removed filter(s) this logbook does not offer: shift' },
    });
    const banners = /** @type {HTMLElement} */ (root().querySelector('.ariel-list-banners'));
    expect(banners.firstElementChild?.id, 'config banner stays first').toBe(
      'config-warning-banner'
    );
    const notice = /** @type {HTMLElement} */ (banners.querySelector('[data-list-notice]'));
    expect(notice.hidden).toBe(false);
    expect(notice.textContent).toContain('does not offer: shift');
  });

  test('the next filter change clears the notice; a filter written with its notice keeps it', async () => {
    await mount();
    const notice = /** @type {HTMLElement} */ (root().querySelector('[data-list-notice]'));
    expect(notice.hidden, 'no notice, no line').toBe(true);
    store.setState({ filter: { author: ['alice'] }, notice: 'dropped: shift' });
    expect(notice.hidden).toBe(false);
    expect(store.getState().notice).toBe('dropped: shift');
    store.setState({ filter: { author: ['bob'] } });
    expect(store.getState().notice).toBeNull();
    expect(notice.hidden).toBe(true);
  });
});

describe('row times and previews', () => {
  /** @returns {Promise<any>} */
  async function components() {
    return import('../../../src/osprey/interfaces/ariel/static/js/components.js');
  }

  test('a row under a day heading shows the time only; the full date is its tooltip', async () => {
    const c = await components();
    const { html } = c.renderDayGroupedRows([entry('E1', '2026-08-25T12:00:00Z')], null, null);
    const host = document.createElement('div');
    host.innerHTML = html;
    const time = /** @type {HTMLElement} */ (host.querySelector('.ariel-row-time'));
    expect(time.textContent).toBe(c.formatTime('2026-08-25T12:00:00Z'));
    expect(time.textContent).not.toContain('2026');
    expect(time.title).toBe(c.formatTimestamp('2026-08-25T12:00:00Z'));
  });

  test('a row with no day heading (search results) keeps the full date', async () => {
    const c = await components();
    const host = document.createElement('div');
    host.innerHTML = c.renderListRow(entry('E1', '2026-08-25T12:00:00Z'));
    expect(host.querySelector('.ariel-row-time')?.textContent).toBe(
      c.formatTimestamp('2026-08-25T12:00:00Z')
    );
  });

  test('a search row with no highlight and no snippet previews the entry text', async () => {
    const c = await components();
    const row = {
      ...entry('E1'),
      snippet: undefined,
      highlights: [],
      raw_text: 'Linac gun cathode replacement\n\nReplaced the thermionic cathode.',
    };
    const host = document.createElement('div');
    host.innerHTML = c.renderListRow(row);
    expect(host.querySelector('.ariel-row-preview')?.textContent).toBe(
      'Replaced the thermionic cathode.'
    );
  });
});
