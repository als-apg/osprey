// @ts-check
/**
 * One artifact listing of the session tree (sidebar-artifact-listing.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-artifact-listing.test.mjs
 *
 * A listing is the artifacts panel's `POST /api/artifacts/session-list` seen
 * through the hub's proxy: one session (or the active key, or Earlier),
 * optionally one form, always `runs: "grouped"`, never the retired `run=`
 * filter. These tests pin the request it builds (through the per-user prefix),
 * the page sequence (a first page replaces and supersedes, the next page
 * appends once, a failure keeps what is held) and the live edits that keep the
 * total honest.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifact-listing.js';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifact-listing.js')} */
let mod;

/**
 * @param {string} id
 * @param {string} timestamp
 * @param {object} [extra]
 */
const entry = (id, timestamp, extra = {}) => ({ id, title: id, timestamp, ...extra });

/** A JSON response. @param {unknown} body @param {number} [status] */
const reply = (body, status = 200) => ({
  ok: status < 400,
  status,
  json: async () => body,
});

/** @type {ReturnType<typeof vi.fn>} */
let fetchMock;
/** @type {string | null} */
let session;
/** @type {import('vitest').Mock<() => void>} */
let onChange;

/**
 * @param {import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifact-listing.js').ListingScope} [scope]
 */
function listing(scope = { session: /** @type {string} */ (session) }) {
  return mod.createListing({ scope, onChange });
}

/** The body of the n-th request. @param {number} n */
function sent(n) {
  return JSON.parse(fetchMock.mock.calls[n][1].body);
}

beforeEach(async () => {
  vi.resetModules();
  mod = await import(MODULE);
  session = 'sess-1';
  onChange = vi.fn();
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  delete window.__OSPREY_PREFIX__;
});

describe('the listing request', () => {
  test('a session, grouped runs, no run filter, a cursor riding along', () => {
    expect(listing().body()).toEqual({ session: 'sess-1', runs: 'grouped' });
    expect(listing({ key: 'k1', form: 'plots' }).body('abc+=')).toEqual({
      key: 'k1',
      form: 'plots',
      runs: 'grouped',
      cursor: 'abc+=',
    });
    expect(JSON.stringify(listing().body())).not.toMatch(/"run"|run_record/);
  });

  test('POSTs to session-list through the per-user prefix', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    fetchMock.mockResolvedValue(reply({ artifacts: [], total: 0 }));
    await listing().loadFirst();
    expect(fetchMock.mock.calls[0][0]).toBe('/u/alice/panel/artifacts/api/artifacts/session-list');
    expect(fetchMock.mock.calls[0][1].method).toBe('POST');
    expect(sent(0)).toEqual({ session: 'sess-1', runs: 'grouped' });
  });
});

describe('the page sequence', () => {
  test('a first page replaces the entries, newest first, with the total and cursor', async () => {
    fetchMock.mockResolvedValue(
      reply({
        artifacts: [entry('b', '2026-10-30T01:00:00Z'), entry('a', '2026-10-30T02:00:00Z')],
        total: 5,
        next_cursor: 'c1',
      })
    );
    const list = listing();
    expect(list.loaded).toBe(false);
    await list.loadFirst();
    expect(list.entries.map((a) => a.id)).toEqual(['a', 'b']);
    expect(list.total).toBe(5);
    expect(list.hasMore).toBe(true);
    expect(list.loaded).toBe(true);
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][1].cache).toBe('no-store');
  });

  test('the next page appends what is new and follows its cursor', async () => {
    fetchMock
      .mockResolvedValueOnce(
        reply({
          artifacts: [entry('c', '2026-10-30T03:00:00Z'), entry('b', '2026-10-30T02:00:00Z')],
          total: 3,
          next_cursor: 'c1',
        })
      )
      .mockResolvedValueOnce(
        reply({
          artifacts: [entry('b', '2026-10-30T02:00:00Z'), entry('a', '2026-10-30T01:00:00Z')],
          total: 3,
          next_cursor: null,
        })
      );
    const list = listing();
    await list.loadFirst();
    await list.loadNext();
    expect(sent(1).cursor).toBe('c1');
    expect(list.entries.map((a) => a.id)).toEqual(['c', 'b', 'a']);
    expect(list.hasMore).toBe(false);
    await list.loadNext();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test('one next page at a time, and none while a first page is in flight', async () => {
    /** @type {(value: unknown) => void} */
    let release = () => {};
    fetchMock.mockResolvedValueOnce(reply({ artifacts: [], total: 9, next_cursor: 'c1' }));
    const list = listing();
    await list.loadFirst();
    fetchMock.mockReturnValueOnce(new Promise((resolve) => (release = resolve)));
    const first = list.loadNext();
    void list.loadNext();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    release(reply({ artifacts: [], total: 9, next_cursor: 'c2' }));
    await first;

    fetchMock.mockReturnValueOnce(new Promise(() => {}));
    void list.loadFirst();
    void list.loadNext();
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  test('a newer first page supersedes an older one still in flight', async () => {
    /** @type {(value: unknown) => void} */
    let releaseOld = () => {};
    fetchMock
      .mockReturnValueOnce(new Promise((resolve) => (releaseOld = resolve)))
      .mockResolvedValueOnce(
        reply({ artifacts: [entry('new', '2026-10-30T02:00:00Z')], total: 1 })
      );
    const list = listing();
    const old = list.loadFirst();
    await list.loadFirst();
    releaseOld(reply({ artifacts: [entry('old', '2026-10-30T01:00:00Z')], total: 1 }));
    await old;
    expect(list.entries.map((a) => a.id)).toEqual(['new']);
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  test('a failed first page keeps what the list held and says so', async () => {
    fetchMock
      .mockResolvedValueOnce(reply({ artifacts: [entry('a', '2026-10-30T01:00:00Z')], total: 1 }))
      .mockResolvedValueOnce(reply({}, 502));
    const list = listing();
    await list.loadFirst();
    await list.loadFirst();
    expect(list.entries.map((a) => a.id)).toEqual(['a']);
    expect(list.failed).toBe(true);
    expect(onChange).toHaveBeenCalledTimes(2);
  });

  test('a stopped listing makes no request and lands none', async () => {
    /** @type {(value: unknown) => void} */
    let release = () => {};
    fetchMock.mockReturnValueOnce(new Promise((resolve) => (release = resolve)));
    const list = listing();
    const pending = list.loadFirst();
    list.stop();
    release(reply({ artifacts: [entry('a', '2026-10-30T01:00:00Z')], total: 1 }));
    await pending;
    await list.loadFirst();
    expect(list.entries).toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe('the live edits', () => {
  /** @returns {Promise<ReturnType<typeof listing>>} */
  async function loadedWith(/** @type {object[]} */ artifacts, total = artifacts.length) {
    fetchMock.mockResolvedValueOnce(reply({ artifacts, total }));
    const list = listing();
    await list.loadFirst();
    return list;
  }

  test('a saved artifact is held newest first and counted once', async () => {
    const list = await loadedWith([entry('a', '2026-10-30T01:00:00Z')]);
    list.add(entry('b', '2026-10-30T02:00:00Z'));
    list.add(entry('b', '2026-10-30T02:00:00Z'));
    expect(list.entries.map((a) => a.id)).toEqual(['b', 'a']);
    expect(list.total).toBe(2);
  });

  test('an artifact fetched on its own is held uncounted until its save event', async () => {
    const list = await loadedWith([entry('a', '2026-10-30T01:00:00Z')], 7);
    list.add(entry('old', '2026-10-01T01:00:00Z'), { counted: false });
    expect(list.total).toBe(7);
    list.add(entry('old', '2026-10-01T01:00:00Z'));
    expect(list.total).toBe(8);
    list.add(entry('old', '2026-10-01T01:00:00Z'));
    expect(list.total).toBe(8);
  });

  test('a run held by its exec_id is replaced by its new head, counted once', async () => {
    const list = await loadedWith([entry('pre', '2026-10-30T01:00:00Z', { exec_id: 'x1' })]);
    list.add(entry('res', '2026-10-30T02:00:00Z', { exec_id: 'x1' }));
    expect(list.entries.map((a) => a.id)).toEqual(['res']);
    expect(list.total).toBe(1);
  });

  test('an update merges into the held entry; an unknown id is not held', async () => {
    const list = await loadedWith([entry('a', '2026-10-30T01:00:00Z')]);
    expect(list.update({ id: 'a', pinned: true })).toBe(true);
    expect(list.find('a')?.pinned).toBe(true);
    expect(list.update({ id: 'nope', pinned: true })).toBe(false);
  });

  test('a delete drops a held entry and its count; an unheld id changes nothing', async () => {
    const list = await loadedWith([entry('a', '2026-10-30T01:00:00Z')], 4);
    expect(list.remove('nope')).toBe(false);
    expect(list.total).toBe(4);
    expect(list.remove('a')).toBe(true);
    expect(list.total).toBe(3);
    list.add(entry('old', '2026-10-01T01:00:00Z'), { counted: false });
    list.remove('old');
    expect(list.total).toBe(3);
  });

  test("an entry held on its own is in the server's total once its page arrives", async () => {
    fetchMock
      .mockResolvedValueOnce(
        reply({ artifacts: [entry('a', '2026-10-30T01:00:00Z')], total: 2, next_cursor: 'c1' })
      )
      .mockResolvedValueOnce(
        reply({ artifacts: [entry('old', '2026-10-01T01:00:00Z')], total: 2, next_cursor: null })
      );
    const list = listing();
    await list.loadFirst();
    list.add(entry('old', '2026-10-01T01:00:00Z'), { counted: false });
    await list.loadNext();
    expect(list.total).toBe(2);
    // The server's total counts it now, so its delete counts down.
    list.remove('old');
    expect(list.total).toBe(1);
  });
});

describe('fetchArtifact', () => {
  test('reads one artifact through the proxy, null when it is gone', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    fetchMock.mockResolvedValueOnce(reply(entry('a/b', '2026-10-30T01:00:00Z')));
    expect((await mod.fetchArtifact('a/b'))?.id).toBe('a/b');
    expect(fetchMock.mock.calls[0][0]).toBe('/u/alice/panel/artifacts/api/artifacts/a%2Fb');
    fetchMock.mockResolvedValueOnce(reply({ detail: 'gone' }, 404));
    expect(await mod.fetchArtifact('x')).toBeNull();
  });
});

describe('fetchFocusedId', () => {
  test('reads the persisted focus through the proxy', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    fetchMock.mockResolvedValueOnce(
      reply({ focused: true, artifact: entry('f1', '2026-10-30T01:00:00Z') })
    );
    expect(await mod.fetchFocusedId()).toBe('f1');
    expect(fetchMock.mock.calls[0][0]).toBe('/u/alice/panel/artifacts/api/focus');
  });

  test("the panel's newest-artifact fallback is no focus, nor is a failure", async () => {
    fetchMock.mockResolvedValueOnce(
      reply({ focused: false, artifact: entry('n1', '2026-10-30T01:00:00Z') })
    );
    expect(await mod.fetchFocusedId()).toBeNull();
    fetchMock.mockResolvedValueOnce(reply({ focused: false, artifact: null }));
    expect(await mod.fetchFocusedId()).toBeNull();
    fetchMock.mockResolvedValueOnce(reply({ detail: 'down' }, 502));
    expect(await mod.fetchFocusedId()).toBeNull();
  });
});

describe('fetchSummary', () => {
  test('POSTs one page of session ids, null on failure', async () => {
    fetchMock.mockResolvedValueOnce(reply({ counts: {}, owned_total: 0 }));
    expect(await mod.fetchSummary(['t1', 't2'])).toMatchObject({ counts: {} });
    expect(fetchMock.mock.calls[0][0]).toBe('/panel/artifacts/api/artifacts/session-summary');
    expect(sent(0)).toEqual({ sessions: ['t1', 't2'] });
    fetchMock.mockResolvedValueOnce(reply({}, 502));
    expect(await mod.fetchSummary(['t1'])).toBeNull();
  });
});
