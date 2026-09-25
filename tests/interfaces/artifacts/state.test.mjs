/**
 * Unit tests for the Artifact Gallery state/fetch/filter layer (state.js).
 *
 * Pure-logic/DOM guard, happy-dom environment (configured globally), `fetch`
 * mocked via vi.stubGlobal — mirrors tests/interfaces/lattice_dashboard/net.test.mjs
 * and tests/interfaces/web_terminal/scaffold-data.test.mjs:
 *   npx vitest run tests/interfaces/artifacts/state.test.mjs
 *
 * Covers accessor correctness (every get/set pair round-trips), fileUrl,
 * the error banner, fetchArtifacts/fetchFocus (success + both failure
 * paths), and getFilteredArtifacts' search/sort combinations over
 * fixture artifacts.
 *
 * NOTE: state.js exports module-singleton state (there's only ever one
 * gallery per page, so a single shared instance is by design). Tests that
 * touch shared state call the relevant setters first, so execution order
 * between tests doesn't matter.
 */

import { test, expect, vi, describe, afterEach } from 'vitest';

import { byId } from '../_support/dom.mjs';
import {
  getArtifacts,
  setArtifacts,
  getSelectedArtifact,
  setSelectedArtifact,
  getFocusedArtifact,
  setFocusedArtifact,
  getCurrentSessionId,
  setCurrentSessionId,
  getShowAllSessions,
  setShowAllSessions,
  fileUrl,
  showErrorBanner,
  hideErrorBanner,
  fetchArtifacts,
  fetchMoreArtifacts,
  fetchFocus,
  getFilteredArtifacts,
  getArtifactTotal,
  hasMoreArtifacts,
  inCurrentScope,
  addArtifact,
  removeArtifact,
  getListSearch,
} from '../../../src/osprey/interfaces/artifacts/static/js/state.js';

afterEach(() => {
  vi.unstubAllGlobals();
  hideErrorBanner();
  document.getElementById('error-banner')?.remove();
});

describe('accessor correctness', () => {
  test('getArtifacts/setArtifacts round-trip the same array reference', () => {
    const list = [{ id: '1' }, { id: '2' }];
    setArtifacts(list);
    expect(getArtifacts()).toBe(list);
  });

  test('getSelectedArtifact/setSelectedArtifact round-trip, including null', () => {
    const a = { id: 'sel-1' };
    setSelectedArtifact(a);
    expect(getSelectedArtifact()).toBe(a);
    setSelectedArtifact(null);
    expect(getSelectedArtifact()).toBeNull();
  });

  test('getFocusedArtifact/setFocusedArtifact round-trip, including null', () => {
    const a = { id: 'focus-1' };
    setFocusedArtifact(a);
    expect(getFocusedArtifact()).toBe(a);
    setFocusedArtifact(null);
    expect(getFocusedArtifact()).toBeNull();
  });

  test('getCurrentSessionId/setCurrentSessionId round-trip, including null', () => {
    setCurrentSessionId('session-abc');
    expect(getCurrentSessionId()).toBe('session-abc');
    setCurrentSessionId(null);
    expect(getCurrentSessionId()).toBeNull();
  });

  test('getShowAllSessions/setShowAllSessions round-trip', () => {
    setShowAllSessions(true);
    expect(getShowAllSessions()).toBe(true);
    setShowAllSessions(false);
    expect(getShowAllSessions()).toBe(false);
  });
});

describe('fileUrl', () => {
  test('builds a /files/{id}/{filename} URL, encoding the filename', () => {
    expect(fileUrl({ id: 'abc123', filename: 'plot one.png' })).toBe('/files/abc123/plot%20one.png');
  });

  test('percent-encodes a hostile id, so no raw path-breakout/query/quote characters survive', () => {
    const url = fileUrl({ id: 'a/../b?x="y"', filename: 'plot.png' });
    const idSegment = url.split('/')[2];
    expect(idSegment).not.toMatch(/[/?"]/);
    expect(url).toBe('/files/a%2F..%2Fb%3Fx%3D%22y%22/plot.png');
  });

  test('is byte-identical for a real 12-hex artifact id (encodeURIComponent is a no-op)', () => {
    expect(fileUrl({ id: '0123456789ab', filename: 'plot.png' })).toBe('/files/0123456789ab/plot.png');
  });
});

describe('error banner', () => {
  test('showErrorBanner creates the banner element on first call and displays the message', () => {
    expect(document.getElementById('error-banner')).toBeNull();
    showErrorBanner('something broke');
    const banner = document.getElementById('error-banner');
    expect(banner).not.toBeNull();
    if (banner === null) throw new Error('unreachable: asserted not-null above');
    expect(banner.textContent).toBe('something broke');
    expect(banner.style.display).toBe('block');
  });

  test('a second showErrorBanner call reuses the existing element', () => {
    showErrorBanner('first message');
    const first = byId('error-banner');
    showErrorBanner('second message');
    const second = byId('error-banner');
    expect(second).toBe(first);
    expect(second.textContent).toBe('second message');
  });

  test('hideErrorBanner hides an existing banner without throwing when none exists', () => {
    showErrorBanner('to be hidden');
    hideErrorBanner();
    expect(byId('error-banner').style.display).toBe('none');

    byId('error-banner').remove();
    expect(() => hideErrorBanner()).not.toThrow();
  });
});

describe('fetchArtifacts', () => {
  test('on success: updates artifacts, hides the error banner, and fires onHealthChange(true)/onArtifactsUpdated', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    showErrorBanner('stale error');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ artifacts: [{ id: 'a1' }] }),
    }));

    const onHealthChange = vi.fn();
    const onArtifactsUpdated = vi.fn();
    await fetchArtifacts({ onHealthChange, onArtifactsUpdated });

    expect(getArtifacts()).toEqual([{ id: 'a1' }]);
    expect(onHealthChange).toHaveBeenCalledWith(true);
    expect(onArtifactsUpdated).toHaveBeenCalledTimes(1);
    expect(byId('error-banner').style.display).toBe('none');
  });

  test('scopes the request to the current session unless showAllSessions is set', async () => {
    setCurrentSessionId('sess-42');
    setShowAllSessions(false);
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
    vi.stubGlobal('fetch', fetchMock);

    await fetchArtifacts();
    expect(fetchMock).toHaveBeenCalledWith('/api/artifacts?session_id=sess-42');

    setShowAllSessions(true);
    await fetchArtifacts();
    expect(fetchMock).toHaveBeenLastCalledWith('/api/artifacts');
  });

  test('on a non-OK response: shows the error banner and fires onHealthChange(false), leaving artifacts untouched', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    setArtifacts([{ id: 'kept' }]);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      text: () => Promise.resolve('boom'),
    }));

    const onHealthChange = vi.fn();
    await fetchArtifacts({ onHealthChange });

    expect(onHealthChange).toHaveBeenCalledWith(false);
    expect(getArtifacts()).toEqual([{ id: 'kept' }]);
    expect(byId('error-banner').textContent).toContain('API error (500)');
  });

  test('on a network failure: shows the error banner with the error message and fires onHealthChange(false)', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));

    const onHealthChange = vi.fn();
    await fetchArtifacts({ onHealthChange });

    expect(onHealthChange).toHaveBeenCalledWith(false);
    expect(byId('error-banner').textContent).toBe('Failed to fetch artifacts: network down');
  });

  test('records the total and the cursor, and defaults the total to the returned length', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ artifacts: [{ id: 'a1' }], total: 40, next_cursor: 'c1' }),
    }));
    await fetchArtifacts();
    expect(getArtifactTotal()).toBe(40);
    expect(hasMoreArtifacts()).toBe(true);

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ artifacts: [{ id: 'a1' }, { id: 'a2' }] }),
    }));
    await fetchArtifacts();
    expect(getArtifactTotal()).toBe(2);
    expect(hasMoreArtifacts()).toBe(false);
  });

  test('carries an escaped search, with and without a session scope', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
    vi.stubGlobal('fetch', fetchMock);

    setCurrentSessionId(null);
    setShowAllSessions(false);
    await fetchArtifacts({ search: 'beam & orbit' });
    expect(fetchMock).toHaveBeenLastCalledWith('/api/artifacts?search=beam%20%26%20orbit');

    setCurrentSessionId('sess/1');
    await fetchArtifacts({ search: 'a+b' });
    expect(fetchMock).toHaveBeenLastCalledWith('/api/artifacts?search=a%2Bb&session_id=sess%2F1');
    setCurrentSessionId(null);
  });

  test('is safe to call with no callbacks at all', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ artifacts: [] }) }));
    await expect(fetchArtifacts()).resolves.toBeUndefined();
  });
});

describe('fetchFocus', () => {
  test('sets focusedArtifact from a successful response', async () => {
    setFocusedArtifact(null);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ artifact: { id: 'focused-1' } }),
    }));

    await fetchFocus();
    expect(getFocusedArtifact()).toEqual({ id: 'focused-1' });
  });

  test('leaves focusedArtifact untouched when the response has no artifact', async () => {
    setFocusedArtifact({ id: 'unchanged' });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({}) }));

    await fetchFocus();
    expect(getFocusedArtifact()).toEqual({ id: 'unchanged' });
  });

  test('is silent (does not throw) on a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));
    await expect(fetchFocus()).resolves.toBeUndefined();
  });
});

describe('getFilteredArtifacts', () => {
  function makeFixtures() {
    return [
      { id: '1', title: 'Beam Profile', filename: 'beam_profile.png', artifact_type: 'plot_png', category: 'visualization', pinned: false, timestamp: '2026-07-01T10:00:00Z' },
      { id: '2', title: 'Channel Values', filename: 'channels.json', artifact_type: 'json', category: 'channel_values', pinned: true, timestamp: '2026-07-03T10:00:00Z' },
      { id: '3', title: 'Lattice Table', filename: 'lattice.html', artifact_type: 'table_html', category: 'visualization', description: 'A summary of magnet strengths', pinned: false, timestamp: '2026-07-02T10:00:00Z' },
      { id: '4', title: 'Old Report', filename: 'report.md', artifact_type: 'markdown', category: 'document', pinned: true, timestamp: '2026-06-30T10:00:00Z' },
    ];
  }

  test('no search returns everything, pinned first then newest-first', () => {
    setArtifacts(makeFixtures());

    const result = getFilteredArtifacts('');
    expect(result.map((a) => a.id)).toEqual(['2', '4', '3', '1']);
  });

  test('search matches title, filename, description, or artifact_type (case-insensitive)', () => {
    setArtifacts(makeFixtures());

    expect(getFilteredArtifacts('beam').map((a) => a.id)).toEqual(['1']);
    expect(getFilteredArtifacts('channels.json').map((a) => a.id)).toEqual(['2']);
    expect(getFilteredArtifacts('magnet strengths').map((a) => a.id)).toEqual(['3']);
    expect(getFilteredArtifacts('markdown').map((a) => a.id)).toEqual(['4']);
  });

  test('an empty search query (default) is treated as no search filter', () => {
    setArtifacts(makeFixtures());
    expect(getFilteredArtifacts().length).toBe(4);
  });

  test('does not mutate the underlying artifacts array', () => {
    const fixtures = makeFixtures();
    setArtifacts(fixtures);

    getFilteredArtifacts('');
    expect(getArtifacts()).toBe(fixtures);
    expect(fixtures.map((a) => a.id)).toEqual(['1', '2', '3', '4']);
  });
});

describe('paging', () => {
  /** @param {any} body */
  function okResponse(body) {
    return { ok: true, json: () => Promise.resolve(body) };
  }

  /** @param {any} firstPage */
  async function loadFirstPage(firstPage) {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResponse(firstPage)));
    await fetchArtifacts();
  }

  test('fetchMoreArtifacts appends the next page, sends the held cursor, and fires onArtifactsUpdated', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }, { id: 'a2' }], total: 3, next_cursor: 'cur 1' });
    const fetchMock = vi.fn().mockResolvedValue(okResponse({ artifacts: [{ id: 'a3' }], total: 3, next_cursor: null }));
    vi.stubGlobal('fetch', fetchMock);

    const onArtifactsUpdated = vi.fn();
    await fetchMoreArtifacts({ onArtifactsUpdated });

    expect(fetchMock).toHaveBeenCalledWith('/api/artifacts?cursor=cur%201');
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2', 'a3']);
    expect(onArtifactsUpdated).toHaveBeenCalledTimes(1);
    expect(hasMoreArtifacts()).toBe(false);
  });

  test('fetchMoreArtifacts is a no-op with no cursor held', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 1, next_cursor: null });
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    await fetchMoreArtifacts();

    expect(fetchMock).not.toHaveBeenCalled();
  });

  test('a second call while a page is in flight issues no second request', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 2, next_cursor: 'c1' });
    /** @type {(v: any) => void} */
    let release = () => {};
    const fetchMock = vi.fn().mockReturnValue(new Promise((r) => { release = r; }));
    vi.stubGlobal('fetch', fetchMock);

    const first = fetchMoreArtifacts();
    await fetchMoreArtifacts();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    release(okResponse({ artifacts: [{ id: 'a2' }], total: 2, next_cursor: null }));
    await first;
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2']);
  });

  test('a next-page call while a first page is in flight issues no request, and the settled page answers it', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 2, next_cursor: 'c1' });
    /** @type {(v: any) => void} */
    let release = () => {};
    const fetchMock = vi.fn().mockReturnValue(new Promise((r) => { release = r; }));
    vi.stubGlobal('fetch', fetchMock);

    const first = fetchArtifacts({ search: 'b' });
    await fetchMoreArtifacts();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    release(okResponse({ artifacts: [{ id: 'b1' }], total: 2, next_cursor: 'cb' }));
    await first;
    expect(getArtifacts().map((a) => a.id)).toEqual(['b1']);

    fetchMock.mockResolvedValue(okResponse({ artifacts: [{ id: 'b2' }], total: 2, next_cursor: null }));
    await fetchMoreArtifacts();
    expect(fetchMock).toHaveBeenLastCalledWith('/api/artifacts?search=b&cursor=cb');
    expect(getArtifacts().map((a) => a.id)).toEqual(['b1', 'b2']);
  });

  test('onArtifactsUpdated may chain the next page, so a short list fills over three or more pages', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 4, next_cursor: 'c1' });
    const pages = [
      { artifacts: [{ id: 'a2' }], total: 4, next_cursor: 'c2' },
      { artifacts: [{ id: 'a3' }], total: 4, next_cursor: 'c3' },
      { artifacts: [{ id: 'a4' }], total: 4, next_cursor: null },
    ];
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(okResponse(pages.shift())));
    vi.stubGlobal('fetch', fetchMock);

    /** @type {Promise<void>[]} */
    const chained = [];
    const callbacks = { onArtifactsUpdated: () => { chained.push(fetchMoreArtifacts(callbacks)); } };
    await fetchMoreArtifacts(callbacks);
    while (chained.length) await chained.shift();

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2', 'a3', 'a4']);
  });

  test('a call while a chained page is in flight issues no request', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 3, next_cursor: 'c1' });
    /** @type {(v: any) => void} */
    let release = () => {};
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(okResponse({ artifacts: [{ id: 'a2' }], total: 3, next_cursor: 'c2' }))
      .mockReturnValueOnce(new Promise((r) => { release = r; }));
    vi.stubGlobal('fetch', fetchMock);

    /** @type {Promise<void>[]} */
    const chained = [];
    const callbacks = { onArtifactsUpdated: () => { chained.push(fetchMoreArtifacts(callbacks)); } };
    await fetchMoreArtifacts(callbacks);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    await fetchMoreArtifacts(callbacks);
    expect(fetchMock).toHaveBeenCalledTimes(2);

    release(okResponse({ artifacts: [{ id: 'a3' }], total: 3, next_cursor: null }));
    while (chained.length) await chained.shift();
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2', 'a3']);
  });

  test('a duplicate id in the appended page is not added twice', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }, { id: 'a2' }], total: 3, next_cursor: 'c1' });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      okResponse({ artifacts: [{ id: 'a2' }, { id: 'a3' }], total: 3, next_cursor: null }),
    ));

    await fetchMoreArtifacts();

    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2', 'a3']);
  });

  test('hasMoreArtifacts follows the cursor, and a failed page leaves it true', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 2, next_cursor: 'c1' });
    expect(hasMoreArtifacts()).toBe(true);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));

    await fetchMoreArtifacts();

    expect(hasMoreArtifacts()).toBe(true);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1']);
  });

  test('inCurrentScope admits untagged and matching entries, refuses a foreign one, admits all with all-sessions on', () => {
    setCurrentSessionId('s1');
    setShowAllSessions(false);
    expect(inCurrentScope({ id: 'x' })).toBe(true);
    expect(inCurrentScope({ id: 'x', session_id: '' })).toBe(true);
    expect(inCurrentScope({ id: 'x', session_id: 's1' })).toBe(true);
    expect(inCurrentScope({ id: 'x', session_id: 's2' })).toBe(false);
    setShowAllSessions(true);
    expect(inCurrentScope({ id: 'x', session_id: 's2' })).toBe(true);
    setShowAllSessions(false);
    setCurrentSessionId(null);
  });

  test('addArtifact returns false while a search is active and leaves the list alone', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResponse({ artifacts: [{ id: 'a1' }], total: 1 })));
    await fetchArtifacts({ search: 'beam' });

    expect(addArtifact({ id: 'new' })).toBe(false);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1']);
    expect(getArtifactTotal()).toBe(1);
  });

  test('addArtifact returns true without adding a foreign-session entry', async () => {
    setCurrentSessionId('s1');
    setShowAllSessions(false);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResponse({ artifacts: [{ id: 'a1' }], total: 1 })));
    await fetchArtifacts();

    expect(addArtifact({ id: 'foreign', session_id: 's2' })).toBe(true);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1']);
    expect(getArtifactTotal()).toBe(1);
    setCurrentSessionId(null);
  });

  test('addArtifact adds, bumps the total, and refuses a duplicate', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 1 });

    expect(addArtifact({ id: 'a2' })).toBe(true);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2']);
    expect(getArtifactTotal()).toBe(2);

    expect(addArtifact({ id: 'a2' })).toBe(true);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2']);
    expect(getArtifactTotal()).toBe(2);
  });

  test('addArtifact holds an entry the total already counts without bumping the total', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 3, next_cursor: 'c1' });

    expect(addArtifact({ id: 'older' }, { counted: true })).toBe(true);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'older']);
    expect(getArtifactTotal()).toBe(3);
  });

  test('an entry held on its own before its save event is counted once when that event arrives', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 3 });

    addArtifact({ id: 'new' }, { counted: true });
    expect(getArtifactTotal()).toBe(3);
    expect(addArtifact({ id: 'new' })).toBe(true);
    expect(getArtifactTotal()).toBe(4);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'new']);
    addArtifact({ id: 'new' });
    expect(getArtifactTotal()).toBe(4);
  });

  test('a fresh first page settles the count of an entry held on its own', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 3 });
    addArtifact({ id: 'new' }, { counted: true });

    await loadFirstPage({ artifacts: [{ id: 'new' }, { id: 'a1' }], total: 4 });
    addArtifact({ id: 'new' });
    expect(getArtifactTotal()).toBe(4);
  });

  test('getListSearch names the search the held list was fetched with', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResponse({ artifacts: [] })));
    await fetchArtifacts({ search: 'orbit' });
    expect(getListSearch()).toBe('orbit');
    await fetchArtifacts();
    expect(getListSearch()).toBe('');
  });

  test('removeArtifact drops the entry, decrements the total, and floors it at zero', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }, { id: 'a2' }], total: 5 });

    removeArtifact('a1');
    expect(getArtifacts().map((a) => a.id)).toEqual(['a2']);
    expect(getArtifactTotal()).toBe(4);

    await loadFirstPage({ artifacts: [], total: 0 });
    setArtifacts([{ id: 'stray' }]);
    removeArtifact('stray');
    expect(getArtifacts()).toEqual([]);
    expect(getArtifactTotal()).toBe(0);
  });

  test('removeArtifact of an entry held uncounted drops it and leaves the total alone', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 3 });
    addArtifact({ id: 'x' }, { counted: true });

    removeArtifact('x');
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1']);
    expect(getArtifactTotal()).toBe(3);
  });
});
