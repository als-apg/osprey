// @ts-check
/**
 * ARIEL list activity (list-activity.js):
 *   npx vitest run tests/interfaces/ariel/list-activity.test.mjs
 *
 * The host relays the agent's `entry_create` / `entry_publish` calls as
 * same-origin `osprey-agent-activity {tool, detail, ts}` messages. Each frame
 * costs one un-reranked side fetch of page 1 of the current view; on a hit the
 * list either refreshes and flashes the row (browsing, at the top, page 1
 * only) or a create adds to the "n new" pill. The selection never moves.
 *
 * Runs under happy-dom with the real list.js and store; only api.js is mocked
 * (the network boundary). Every case resets modules and imports the store, the
 * list and list-activity from the same reset, and stops list-activity after,
 * so no listener from an earlier case answers a later frame.
 */

import { test, expect, beforeEach, afterEach, describe, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    searchApi: { ...actual.searchApi, search: vi.fn() },
    entriesApi: { ...actual.entriesApi, list: vi.fn(), get: vi.fn() },
    draftsApi: { ...actual.draftsApi, list: vi.fn() },
  };
});

const JS = '../../../src/osprey/interfaces/ariel/static/js';

/** Two plain modes plus a reranking one, as /api/capabilities sends them. */
const CAPABILITIES = /** @type {any} */ ({
  categories: {
    direct: {
      label: 'Direct',
      modes: [
        { name: 'keyword', label: 'Keyword', parameters: [] },
        {
          name: 'semantic',
          label: 'Semantic',
          parameters: [{ name: 'rerank', label: 'Rerank', type: 'bool', default: true }],
        },
      ],
    },
  },
  default_mode: 'keyword',
  shared_parameters: [],
  vocabulary: { enabled: false, concepts: 0, expand_by_default: false },
});

/** @type {any} */ let api;
/** @type {any} */ let store;
/** @type {any} */ let activity;
/** @type {(() => void)|null} */ let stop = null;

/**
 * @param {string} id
 * @returns {any}
 */
function entry(id) {
  return {
    entry_id: id,
    timestamp: '2026-08-25T12:00:00Z',
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
 * @param {string[]} ids
 * @param {string|null} [next_cursor]
 */
function listPage(ids, next_cursor = null) {
  const entries = ids.map(entry);
  return { entries, total: entries.length, page: null, page_size: 20, total_pages: 1, next_cursor };
}

/** @param {string[]} ids */
function searchResponse(ids) {
  const entries = ids.map(entry);
  return {
    search_modes_used: ['keyword'],
    execution_time_ms: 3,
    total: entries.length,
    total_results: entries.length,
    total_exact: true,
    has_more: false,
    entries,
    diagnostics: [],
    expanded_terms: [],
  };
}

async function flush() {
  for (let i = 0; i < 4; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Mount the list and list-activity from one fresh module graph.
 * @param {{state?: any, firstPage?: any, search?: any}} [opts]
 */
async function mount({ state = {}, firstPage = listPage(['e2', 'e1']), search } = {}) {
  api = await import(`${JS}/api.js`);
  store = await import(`${JS}/store.js`);
  const list = await import(`${JS}/list.js`);
  activity = await import(`${JS}/list-activity.js`);
  store.setState(state);
  vi.mocked(api.entriesApi.list).mockResolvedValue(firstPage);
  vi.mocked(api.searchApi.search).mockResolvedValue(search ?? searchResponse(['e2', 'e1']));
  vi.mocked(api.draftsApi.list).mockResolvedValue({ drafts: [] });
  list.initList(root(), { capabilities: CAPABILITIES });
  await flush();
  stop = activity.initListActivity({ capabilities: CAPABILITIES });
  vi.mocked(api.entriesApi.list).mockClear();
  vi.mocked(api.searchApi.search).mockClear();
}

/** @returns {HTMLElement} */
function root() {
  return /** @type {HTMLElement} */ (document.getElementById('ariel-list'));
}

/** @returns {HTMLElement} */
function body() {
  return /** @type {HTMLElement} */ (document.getElementById('ariel-list-body'));
}

/** @returns {HTMLButtonElement} */
function pill() {
  return /** @type {HTMLButtonElement} */ (root().querySelector('[data-new-pill]'));
}

/** @returns {string} The pill's text, or '' while it is hidden. */
function pillText() {
  return pill().hidden ? '' : (pill().textContent ?? '');
}

/** @param {string} id */
function row(id) {
  return /** @type {HTMLElement|null} */ (body().querySelector(`[data-entry-id="${id}"]`));
}

/**
 * Post one relayed frame, as the host does.
 * @param {string} tool
 * @param {string} detail
 * @param {{origin?: string, type?: string}} [opts]
 */
function frame(
  tool,
  detail,
  { origin = window.location.origin, type = 'osprey-agent-activity' } = {}
) {
  window.dispatchEvent(
    new MessageEvent('message', { data: { type, tool, detail, ts: 1 }, origin })
  );
}

/** @param {number} px */
function scrollTo(px) {
  body().scrollTop = px;
}

/** Page-1 side fetches and refreshes alike: /api/entries calls since mount. */
function listCalls() {
  return vi.mocked(api.entriesApi.list).mock.calls.length;
}

/**
 * A promise with its resolver.
 * @returns {{promise: Promise<any>, resolve: (v: any) => void}}
 */
function deferred() {
  /** @type {(v: any) => void} */
  let resolve = () => {};
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
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
  stop?.();
  stop = null;
  vi.restoreAllMocks();
  document.documentElement.removeAttribute('data-facility-timezone');
});

describe('entry_create', () => {
  test('a matching create while scrolled gives a pill of 1, without refreshing', async () => {
    await mount();
    scrollTo(300);
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['e3', 'e2', 'e1']));
    frame('entry_create', 'e3');
    await flush();

    expect(pillText()).toBe('1 new');
    expect(listCalls()).toBe(1); // the side fetch only
    expect(row('e3')).toBeNull(); // never spliced in
    expect(body().scrollTop).toBe(300);
  });

  test('the side fetch is page 1 of the current view, through api.js', async () => {
    await mount({ state: { filter: { author: ['bob'] } } });
    frame('entry_create', 'e3');
    await flush();
    const args = vi.mocked(api.entriesApi.list).mock.calls[0][0];
    expect(args).toMatchObject({ pageSize: 20, view: 'row', filters: { author: ['bob'] } });
    expect(args.cursor).toBeUndefined();
  });

  test('a non-matching create changes nothing', async () => {
    await mount();
    scrollTo(300);
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['e2', 'e1']));
    frame('entry_create', 'e9');
    await flush();

    expect(pillText()).toBe('');
    expect(listCalls()).toBe(1);
    expect(body().scrollTop).toBe(300);
  });

  test('a matching create at the top of page 1 refreshes and flashes the row', async () => {
    await mount();
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['e3', 'e2', 'e1']));
    frame('entry_create', 'e3');
    await flush();

    expect(listCalls()).toBe(2); // side fetch, then refresh()
    expect(row('e3')?.classList.contains('agent-flash')).toBe(true);
    expect(pillText()).toBe('');
  });

  test('past page 1 a hit only bumps the pill, even at the top', async () => {
    await mount({ firstPage: listPage(['e2'], 'c1') });
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['e1']));
    /** @type {HTMLElement} */ (body().querySelector('[data-load-more]')).click();
    await flush();
    vi.mocked(api.entriesApi.list).mockClear();

    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['e3', 'e2']));
    frame('entry_create', 'e3');
    await flush();
    expect(pillText()).toBe('1 new');
    expect(listCalls()).toBe(1);
  });

  test('three creates give a pill of 3 with at most two fetches', async () => {
    await mount();
    scrollTo(300);
    const first = deferred();
    const second = deferred();
    vi.mocked(api.entriesApi.list)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    frame('entry_create', 'e3');
    frame('entry_create', 'e4');
    frame('entry_create', 'e5');
    await flush();
    first.resolve(listPage(['e3', 'e2', 'e1']));
    await flush();
    second.resolve(listPage(['e5', 'e4', 'e3', 'e2', 'e1']));
    await flush();

    expect(listCalls()).toBeLessThanOrEqual(2);
    expect(pillText()).toBe('3 new');
  });

  test('clicking the pill scrolls to the top, refreshes and clears it', async () => {
    await mount();
    scrollTo(300);
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['e3', 'e2', 'e1']));
    frame('entry_create', 'e3');
    await flush();
    pill().click();
    await flush();

    expect(body().scrollTop).toBe(0);
    expect(pillText()).toBe('');
    expect(listCalls()).toBe(2);
    expect(row('e3')).not.toBeNull();
  });
});

describe('an entry submitted from this panel', () => {
  /** @param {string} entryId */
  function submitted(entryId) {
    window.dispatchEvent(new CustomEvent('ariel:entry-created', { detail: { entryId } }));
  }

  test('at the top of page 1 it refreshes the list, updates the count and flashes the row', async () => {
    await mount();
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['e3', 'e2', 'e1']));
    submitted('e3');
    await flush();

    expect(listCalls()).toBe(2); // side fetch, then refresh()
    expect(row('e3')?.classList.contains('agent-flash')).toBe(true);
    expect(body().querySelector('[data-list-total]')?.textContent).toBe('3');
    expect(pillText()).toBe('');
  });

  test('while scrolled it counts on the n-new pill, like an agent create', async () => {
    await mount();
    scrollTo(300);
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['e3', 'e2', 'e1']));
    submitted('e3');
    await flush();

    expect(pillText()).toBe('1 new');
  });
});

describe('entry_publish', () => {
  test('a publish at the top flashes the row and leaves the pill alone', async () => {
    await mount();
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['f1', 'e2', 'e1']));
    frame('entry_publish', 'f1');
    await flush();

    expect(listCalls()).toBe(2);
    expect(row('f1')?.classList.contains('agent-flash')).toBe(true);
    expect(pillText()).toBe('');
  });

  test('a publish while scrolled changes nothing', async () => {
    await mount();
    scrollTo(300);
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['f1', 'e2', 'e1']));
    frame('entry_publish', 'f1');
    await flush();

    expect(pillText()).toBe('');
    expect(listCalls()).toBe(1);
    expect(row('f1')).toBeNull();
    expect(body().scrollTop).toBe(300);
  });
});

describe('selection', () => {
  test('the selection is stable through a refresh and a pill bump', async () => {
    await mount({ state: { sel: 'e1' } });
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['e3', 'e2', 'e1']));
    frame('entry_create', 'e3');
    await flush();
    expect(store.getState().sel).toBe('e1');
    expect(row('e1')?.getAttribute('aria-selected')).toBe('true');
    expect(row('e3')?.getAttribute('aria-selected')).toBe('false');

    scrollTo(300);
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['e4', 'e3', 'e2', 'e1']));
    frame('entry_create', 'e4');
    await flush();
    expect(store.getState().sel).toBe('e1');
    expect(row('e1')?.getAttribute('aria-selected')).toBe('true');
  });
});

describe('views', () => {
  test('a search view sends page 1 un-reranked, and a hit only bumps the pill', async () => {
    await mount({ state: { query: 'rf trip', mode: 'semantic' } });
    vi.mocked(api.searchApi.search).mockResolvedValue(searchResponse(['e3', 'e2']));
    frame('entry_create', 'e3');
    await flush();

    expect(api.searchApi.search).toHaveBeenCalledTimes(1);
    const args = vi.mocked(api.searchApi.search).mock.calls[0][0];
    expect(args).toMatchObject({ query: 'rf trip', mode: 'semantic', offset: 0, maxResults: 20 });
    expect(args.advancedParams.rerank).toBe(false);
    expect(pillText()).toBe('1 new');
    expect(row('e3')).toBeNull();
  });

  test('Drafts ignores frames', async () => {
    await mount({ state: { mailbox: 'drafts' } });
    frame('entry_create', 'e3');
    await flush();
    expect(listCalls()).toBe(0);
    expect(pillText()).toBe('');
  });

  test('Mine without an author ignores frames; with one it filters by author', async () => {
    await mount({ state: { mailbox: 'mine', mineAuthor: null } });
    frame('entry_create', 'e3');
    await flush();
    expect(listCalls()).toBe(0);

    store.setState({ mineAuthor: 'alice' });
    await flush();
    vi.mocked(api.entriesApi.list).mockClear();
    frame('entry_create', 'e3');
    await flush();
    expect(vi.mocked(api.entriesApi.list).mock.calls[0][0].filters).toEqual({
      author: ['alice'],
    });
  });

  test('a view change while a fetch is in flight drops its answer and the pill', async () => {
    await mount();
    scrollTo(300);
    const pending = deferred();
    vi.mocked(api.entriesApi.list).mockReturnValueOnce(pending.promise);
    frame('entry_create', 'e3');
    await flush();
    vi.mocked(api.entriesApi.list).mockResolvedValue(listPage(['e2']));
    store.setState({ filter: { author: ['bob'] } });
    pending.resolve(listPage(['e3', 'e2']));
    await flush();
    expect(pillText()).toBe('');
  });

  test('other tools, other message types and other origins are ignored', async () => {
    await mount();
    frame('entry_open', 'e3');
    frame('entry_create', 'e3', { type: 'something-else' });
    frame('entry_create', 'e3', { origin: 'https://evil.example' });
    await flush();
    expect(listCalls()).toBe(0);
  });
});
