// @ts-check
/**
 * The ARIEL Drafts mailbox (drafts-list.js):
 *   npx vitest run tests/interfaces/ariel/drafts-list.test.mjs
 *
 * Driven through the list pane (list.js) the way the operator drives it: the
 * rail writes `mailbox:'drafts'`, the list hands its body to `renderDrafts`.
 * Rows show subject, author, staged-by and age, newest first, or an empty
 * state; a row click writes `compose:{draft:id}`. A `draftsRev` change
 * refetches while Drafts is shown. A drafts response paints only while it is
 * the latest drafts fetch, the mailbox is still Drafts and the list has
 * started no other run.
 *
 * Runs under happy-dom. Only api.js is mocked (the network boundary); every
 * case resets modules so the store, list and drafts list share one graph.
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
const DRAFTS_PATH = '../../../src/osprey/interfaces/ariel/static/js/drafts-list.js';

/** @type {any} */
let api;
/** @type {any} */
let store;
/** @type {any} */
let list;
/** @type {any} */
let drafts;

const NOW = Date.parse('2026-08-25T12:00:00Z');

/**
 * @param {string} id
 * @param {number} minutesAgo
 * @param {any} [extra]
 * @returns {any}
 */
function draft(id, minutesAgo, extra = {}) {
  return {
    draft_id: id,
    subject: `subject of ${id}`,
    author: 'alice',
    logbook: 'ops',
    created_at: new Date(NOW - minutesAgo * 60000).toISOString(),
    staged_by: 'agent-7',
    attachment_count: 0,
    ...extra,
  };
}

/** @returns {{promise: Promise<any>, resolve: (v: any) => void, reject: (e: any) => void}} */
function deferred() {
  /** @type {(v: any) => void} */
  let resolve = () => {};
  /** @type {(e: any) => void} */
  let reject = () => {};
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function flush() {
  await new Promise((resolve) => setTimeout(resolve, 0));
  await new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * @param {{state?: any, listed?: any}} [opts]
 */
async function mount({ state = { mailbox: 'drafts' }, listed = { drafts: [] } } = {}) {
  api = await import(API_PATH);
  store = await import(STORE_PATH);
  list = await import(LIST_PATH);
  drafts = await import(DRAFTS_PATH);
  vi.mocked(api.entriesApi.list).mockResolvedValue({
    entries: [{ entry_id: 'e1', timestamp: '2026-08-25T11:00:00Z', subject: 'an entry' }],
    total: 1,
    page: null,
    page_size: 20,
    total_pages: 1,
    next_cursor: null,
  });
  vi.mocked(api.searchApi.search).mockResolvedValue({ entries: [], total: 0 });
  vi.mocked(api.draftsApi.list).mockResolvedValue(listed);
  store.setState(state);
  list.initList(document.getElementById('ariel-list'), { capabilities: null });
  await flush();
}

/** @returns {HTMLElement} */
function body() {
  return /** @type {HTMLElement} */ (document.getElementById('ariel-list-body'));
}

/** @returns {HTMLElement[]} */
function rows() {
  return Array.from(body().querySelectorAll('[data-draft-id]'));
}

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  vi.useFakeTimers({ toFake: ['Date'] });
  vi.setSystemTime(NOW);
  localStorage.clear();
  history.replaceState(null, '', '/');
  document.documentElement.setAttribute('data-facility-timezone', 'UTC');
  document.body.innerHTML = '<section id="ariel-list"></section>';
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  history.replaceState(null, '', '/');
  document.documentElement.removeAttribute('data-facility-timezone');
});

describe('rows', () => {
  test('newest first, with subject, author, staged-by and age', async () => {
    await mount({
      listed: {
        drafts: [draft('old', 180), draft('new', 5, { attachment_count: 2 }), draft('mid', 60)],
      },
    });
    expect(rows().map((r) => r.dataset.draftId)).toEqual(['new', 'mid', 'old']);
    const first = rows()[0];
    expect(first.textContent).toContain('subject of new');
    expect(first.textContent).toContain('alice');
    expect(first.textContent).toContain('staged by agent-7');
    expect(first.textContent).toContain('5m ago');
    expect(rows()[2].textContent).toContain('3h ago');
    expect(first.querySelector('[data-attachment-count="2"]')).not.toBeNull();
    expect(body().getAttribute('aria-busy')).toBe('false');
  });

  test('rows never carry data-entry-id, so list selection and j/k skip them', async () => {
    await mount({ listed: { drafts: [draft('d1', 1)] } });
    expect(rows()).toHaveLength(1);
    expect(body().querySelector('[data-entry-id]')).toBeNull();
  });

  test('every value is escaped', async () => {
    const evil = '<img src=x onerror="alert(1)">';
    await mount({
      listed: { drafts: [draft('d"1', 1, { subject: evil, author: evil, staged_by: evil })] },
    });
    expect(body().querySelector('img')).toBeNull();
    expect(rows()[0].dataset.draftId).toBe('d"1');
    expect(rows()[0].textContent).toContain(evil);
  });

  test('no drafts: an empty state', async () => {
    await mount();
    expect(rows()).toHaveLength(0);
    expect(body().textContent).toContain(drafts.DRAFTS_NONE.title);
  });

  test('a failed listing shows an error state', async () => {
    api = await import(API_PATH);
    vi.mocked(api.draftsApi.list).mockRejectedValue(new Error('boom'));
    store = await import(STORE_PATH);
    list = await import(LIST_PATH);
    vi.mocked(api.entriesApi.list).mockResolvedValue({ entries: [], next_cursor: null });
    store.setState({ mailbox: 'drafts' });
    list.initList(document.getElementById('ariel-list'), { capabilities: null });
    await flush();
    expect(body().textContent).toContain('Drafts Failed');
    expect(body().getAttribute('aria-busy')).toBe('false');
  });
});

describe('selection', () => {
  test('a row click writes compose:{draft:id} and leaves sel alone', async () => {
    await mount({ state: { mailbox: 'drafts', sel: 'e9' }, listed: { drafts: [draft('d1', 1)] } });
    rows()[0]
      .querySelector('.ariel-row-subject')
      ?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(store.getState().compose).toEqual({ draft: 'd1' });
    expect(store.getState().sel).toBe('e9');
  });

  test('the draft open in the sheet is marked selected', async () => {
    await mount({
      state: { mailbox: 'drafts', compose: { draft: 'd2' } },
      listed: { drafts: [draft('d1', 1), draft('d2', 2)] },
    });
    expect(rows().map((r) => r.getAttribute('aria-selected'))).toEqual(['false', 'true']);
  });
});

describe('refetching', () => {
  test('a draftsRev change refetches while Drafts is shown', async () => {
    await mount({ listed: { drafts: [draft('d1', 1), draft('d2', 2)] } });
    expect(api.draftsApi.list).toHaveBeenCalledTimes(1);
    vi.mocked(api.draftsApi.list).mockResolvedValue({ drafts: [draft('d2', 2)] });
    store.setState({ draftsRev: store.getState().draftsRev + 1 });
    await flush();
    expect(api.draftsApi.list).toHaveBeenCalledTimes(2);
    expect(rows().map((r) => r.dataset.draftId)).toEqual(['d2']);
  });

  test('a draftsRev change outside Drafts fetches nothing', async () => {
    await mount({ state: { mailbox: 'all' } });
    store.setState({ draftsRev: store.getState().draftsRev + 1 });
    await flush();
    expect(api.draftsApi.list).not.toHaveBeenCalled();

    store.setState({ mailbox: 'drafts' });
    await flush();
    store.setState({ mailbox: 'all' });
    await flush();
    vi.mocked(api.draftsApi.list).mockClear();
    store.setState({ draftsRev: store.getState().draftsRev + 1 });
    await flush();
    expect(api.draftsApi.list).not.toHaveBeenCalled();
  });

  test('entering Drafts again refetches', async () => {
    await mount();
    store.setState({ mailbox: 'all' });
    await flush();
    store.setState({ mailbox: 'drafts' });
    await flush();
    expect(api.draftsApi.list).toHaveBeenCalledTimes(2);
  });
});

describe('the guard', () => {
  test('leave Drafts, then a late drafts response paints nothing', async () => {
    await mount({ state: { mailbox: 'all' } });
    const late = deferred();
    vi.mocked(api.draftsApi.list).mockReturnValue(late.promise);
    store.setState({ mailbox: 'drafts' });
    store.setState({ mailbox: 'all' });
    await flush();
    const before = body().innerHTML;
    late.resolve({ drafts: [draft('d1', 1)] });
    await flush();
    expect(rows()).toHaveLength(0);
    expect(body().innerHTML).toBe(before);
    expect(body().querySelector('[data-entry-id="e1"]')).not.toBeNull();
  });

  test('a late failure after leaving Drafts paints nothing either', async () => {
    await mount({ state: { mailbox: 'all' } });
    const late = deferred();
    vi.mocked(api.draftsApi.list).mockReturnValue(late.promise);
    store.setState({ mailbox: 'drafts' });
    store.setState({ mailbox: 'all' });
    await flush();
    late.reject(new Error('boom'));
    await flush();
    expect(body().textContent).not.toContain('Drafts Failed');
  });

  test('an older drafts fetch answering after a newer one paints nothing', async () => {
    await mount({ state: { mailbox: 'all' } });
    const first = deferred();
    const second = deferred();
    vi.mocked(api.draftsApi.list)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    store.setState({ mailbox: 'drafts' });
    store.setState({ draftsRev: store.getState().draftsRev + 1 });
    second.resolve({ drafts: [draft('fresh', 1)] });
    await flush();
    first.resolve({ drafts: [draft('stale', 1)] });
    await flush();
    expect(rows().map((r) => r.dataset.draftId)).toEqual(['fresh']);
  });

  test('a list refresh started since (still in Drafts) wins over the earlier fetch', async () => {
    await mount({ state: { mailbox: 'all' } });
    const first = deferred();
    vi.mocked(api.draftsApi.list)
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce({ drafts: [draft('fresh', 1)] });
    store.setState({ mailbox: 'drafts' });
    store.setState({ query: 'quench' });
    await flush();
    first.resolve({ drafts: [draft('stale', 1)] });
    await flush();
    expect(rows().map((r) => r.dataset.draftId)).toEqual(['fresh']);
  });
});
