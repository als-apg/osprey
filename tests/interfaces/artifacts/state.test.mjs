/**
 * Unit tests for the Artifact Gallery state/fetch/filter layer (state.js).
 *
 * Pure-logic/DOM guard, happy-dom environment (configured globally), `fetch`
 * mocked via vi.stubGlobal — mirrors tests/interfaces/web_terminal/scaffold/data.test.mjs:
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
  hasMoreArtifacts,
  inCurrentScope,
  liveVerdict,
  followsFocus,
  getHeldTranscript,
  addArtifact,
  removeArtifact,
  getPickedIds,
  setPicked,
  togglePicked,
  clearPicked,
  getPickAnchor,
  setPickAnchor,
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
    expect(fileUrl({ id: 'abc123', filename: 'plot one.png' })).toBe(
      '/files/abc123/plot%20one.png'
    );
  });

  test('percent-encodes a hostile id, so no raw path-breakout/query/quote characters survive', () => {
    const url = fileUrl({ id: 'a/../b?x="y"', filename: 'plot.png' });
    const idSegment = url.split('/')[2];
    expect(idSegment).not.toMatch(/[/?"]/);
    expect(url).toBe('/files/a%2F..%2Fb%3Fx%3D%22y%22/plot.png');
  });

  test('is byte-identical for a real 12-hex artifact id (encodeURIComponent is a no-op)', () => {
    expect(fileUrl({ id: '0123456789ab', filename: 'plot.png' })).toBe(
      '/files/0123456789ab/plot.png'
    );
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
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ artifacts: [{ id: 'a1' }] }),
      })
    );

    const onHealthChange = vi.fn();
    const onArtifactsUpdated = vi.fn();
    await fetchArtifacts({ onHealthChange, onArtifactsUpdated });

    expect(getArtifacts()).toEqual([{ id: 'a1' }]);
    expect(onHealthChange).toHaveBeenCalledWith(true);
    expect(onArtifactsUpdated).toHaveBeenCalledTimes(1);
    expect(byId('error-banner').style.display).toBe('none');
  });

  test('in the session view, posts the gallery body to session-list and holds the transcript', async () => {
    setCurrentSessionId('sess-42');
    setShowAllSessions(false);
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ artifacts: [], transcript: 'T-1', ledger: 'ok' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await fetchArtifacts();
    expect(fetchMock).toHaveBeenCalledWith('/api/artifacts/session-list', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: 'sess-42', search: '', cursor: null, runs: 'grouped' }),
    });
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      key: 'sess-42',
      search: '',
      cursor: null,
      runs: 'grouped',
    });
    expect(getHeldTranscript()).toBe('T-1');

    setShowAllSessions(true);
    await fetchArtifacts();
    expect(fetchMock).toHaveBeenLastCalledWith('/api/artifacts');
    expect(getHeldTranscript()).toBeNull();
    setShowAllSessions(false);
    setCurrentSessionId(null);
  });

  test('a key the ledger has no row for holds a null transcript', async () => {
    setCurrentSessionId('sess-unseen');
    setShowAllSessions(false);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ artifacts: [], transcript: null, ledger: 'ok' }),
      })
    );
    await fetchArtifacts();
    expect(getHeldTranscript()).toBeNull();
    setCurrentSessionId(null);
  });

  test('with no session id, the page lists all sessions through GET /api/artifacts', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ ok: true, json: () => Promise.resolve({ artifacts: [{ id: 'w1' }] }) });
    vi.stubGlobal('fetch', fetchMock);

    await fetchArtifacts();
    expect(fetchMock).toHaveBeenCalledWith('/api/artifacts');
    expect(getHeldTranscript()).toBeNull();
    expect(getArtifacts()).toEqual([{ id: 'w1' }]);
  });

  test('on a non-OK response: shows the error banner and fires onHealthChange(false), leaving artifacts untouched', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    setArtifacts([{ id: 'kept' }]);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        text: () => Promise.resolve('boom'),
      })
    );

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

  test('records the cursor', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ artifacts: [{ id: 'a1' }], total: 40, next_cursor: 'c1' }),
      })
    );
    await fetchArtifacts();
    expect(hasMoreArtifacts()).toBe(true);

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ artifacts: [{ id: 'a1' }, { id: 'a2' }] }),
      })
    );
    await fetchArtifacts();
    expect(hasMoreArtifacts()).toBe(false);
  });

  test('carries an escaped search, with and without a session scope', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
    vi.stubGlobal('fetch', fetchMock);

    setCurrentSessionId(null);
    setShowAllSessions(false);
    await fetchArtifacts({ search: 'beam & orbit' });
    expect(fetchMock).toHaveBeenLastCalledWith('/api/artifacts?search=beam%20%26%20orbit');

    setCurrentSessionId('sess/1');
    await fetchArtifacts({ search: 'a+b' });
    const [url, init] = fetchMock.mock.calls[fetchMock.mock.calls.length - 1];
    expect(url).toBe('/api/artifacts/session-list');
    expect(JSON.parse(init.body)).toEqual({
      key: 'sess/1',
      search: 'a+b',
      cursor: null,
      runs: 'grouped',
    });

    setShowAllSessions(true);
    await fetchArtifacts({ search: 'a+b' });
    expect(fetchMock).toHaveBeenLastCalledWith('/api/artifacts?search=a%2Bb');
    setShowAllSessions(false);
    setCurrentSessionId(null);
  });

  test('is safe to call with no callbacks at all', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ artifacts: [] }) })
    );
    await expect(fetchArtifacts()).resolves.toBeUndefined();
  });
});

describe('fetchFocus', () => {
  test('sets focusedArtifact from a successful response', async () => {
    setFocusedArtifact(null);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ artifact: { id: 'focused-1' } }),
      })
    );

    await fetchFocus();
    expect(getFocusedArtifact()).toEqual({ id: 'focused-1' });
  });

  test('leaves focusedArtifact untouched when the response has no artifact', async () => {
    setFocusedArtifact({ id: 'unchanged' });
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({}) })
    );

    await fetchFocus();
    expect(getFocusedArtifact()).toEqual({ id: 'unchanged' });
  });

  test('is silent (does not throw) on a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));
    await expect(fetchFocus()).resolves.toBeUndefined();
  });

  test("the session view reads its key's focus; the all-sessions view reads any", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ ok: true, json: () => Promise.resolve({ artifact: null }) });
    vi.stubGlobal('fetch', fetchMock);

    setCurrentSessionId('key/1');
    setShowAllSessions(false);
    await fetchFocus();
    expect(fetchMock).toHaveBeenLastCalledWith('/api/focus?session_key=key%2F1');

    setShowAllSessions(true);
    await fetchFocus();
    expect(fetchMock).toHaveBeenLastCalledWith('/api/focus');

    setShowAllSessions(false);
    setCurrentSessionId(null);
    await fetchFocus();
    expect(fetchMock).toHaveBeenLastCalledWith('/api/focus');
  });
});

describe('getFilteredArtifacts', () => {
  function makeFixtures() {
    return [
      {
        id: '1',
        title: 'Beam Profile',
        filename: 'beam_profile.png',
        artifact_type: 'plot_png',
        category: 'visualization',
        pinned: false,
        timestamp: '2026-07-01T10:00:00Z',
      },
      {
        id: '2',
        title: 'Channel Values',
        filename: 'channels.json',
        artifact_type: 'json',
        category: 'channel_values',
        pinned: true,
        timestamp: '2026-07-03T10:00:00Z',
      },
      {
        id: '3',
        title: 'Lattice Table',
        filename: 'lattice.html',
        artifact_type: 'table_html',
        category: 'visualization',
        description: 'A summary of magnet strengths',
        pinned: false,
        timestamp: '2026-07-02T10:00:00Z',
      },
      {
        id: '4',
        title: 'Old Report',
        filename: 'report.md',
        artifact_type: 'markdown',
        category: 'document',
        pinned: true,
        timestamp: '2026-06-30T10:00:00Z',
      },
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
    await loadFirstPage({
      artifacts: [{ id: 'a1' }, { id: 'a2' }],
      total: 3,
      next_cursor: 'cur 1',
    });
    const fetchMock = vi
      .fn()
      .mockResolvedValue(okResponse({ artifacts: [{ id: 'a3' }], total: 3, next_cursor: null }));
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
    const fetchMock = vi.fn().mockReturnValue(
      new Promise((r) => {
        release = r;
      })
    );
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
    const fetchMock = vi.fn().mockReturnValue(
      new Promise((r) => {
        release = r;
      })
    );
    vi.stubGlobal('fetch', fetchMock);

    const first = fetchArtifacts({ search: 'b' });
    await fetchMoreArtifacts();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    release(okResponse({ artifacts: [{ id: 'b1' }], total: 2, next_cursor: 'cb' }));
    await first;
    expect(getArtifacts().map((a) => a.id)).toEqual(['b1']);

    fetchMock.mockResolvedValue(
      okResponse({ artifacts: [{ id: 'b2' }], total: 2, next_cursor: null })
    );
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
    const callbacks = {
      onArtifactsUpdated: () => {
        chained.push(fetchMoreArtifacts(callbacks));
      },
    };
    await fetchMoreArtifacts(callbacks);
    while (chained.length) await chained.shift();

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2', 'a3', 'a4']);
  });

  test('a call while a chained page is in flight issues no request', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 3, next_cursor: 'c1' });
    /** @type {(v: any) => void} */
    let release = () => {};
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(okResponse({ artifacts: [{ id: 'a2' }], total: 3, next_cursor: 'c2' }))
      .mockReturnValueOnce(
        new Promise((r) => {
          release = r;
        })
      );
    vi.stubGlobal('fetch', fetchMock);

    /** @type {Promise<void>[]} */
    const chained = [];
    const callbacks = {
      onArtifactsUpdated: () => {
        chained.push(fetchMoreArtifacts(callbacks));
      },
    };
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
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          okResponse({ artifacts: [{ id: 'a2' }, { id: 'a3' }], total: 3, next_cursor: null })
        )
    );

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

  /**
   * Load a session-view first page for key `key` whose answer names `transcript`.
   * @param {string} key @param {string|null} transcript @param {any[]} [list]
   */
  async function loadSessionView(key, transcript, list = []) {
    setCurrentSessionId(key);
    setShowAllSessions(false);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(okResponse({ artifacts: list, transcript, ledger: 'ok' }))
    );
    await fetchArtifacts();
  }

  test('inCurrentScope admits what the held transcript owns, and everything with all-sessions on', async () => {
    await loadSessionView('k1', 'T1');
    expect(inCurrentScope({ id: 'x', owner: 'T1', operator_visible: true })).toBe(true);
    expect(inCurrentScope({ id: 'x', owner: 'T1', operator_visible: false })).toBe(false);
    expect(inCurrentScope({ id: 'x', owner: 'T2', operator_visible: true })).toBe(false);
    expect(inCurrentScope({ id: 'x', owner: null, operator_visible: true })).toBe(false);
    // The key alone does not admit: the transcript that owns it does.
    expect(inCurrentScope({ id: 'x', session_id: 'k1', owner: 'T2', operator_visible: true })).toBe(
      false
    );
    setShowAllSessions(true);
    expect(inCurrentScope({ id: 'x', owner: 'T2' })).toBe(true);
    setShowAllSessions(false);
    setCurrentSessionId(null);
    expect(inCurrentScope({ id: 'x', owner: 'T2' })).toBe(true);
  });

  test('a held null transcript admits nothing, not even an unowned artifact', async () => {
    await loadSessionView('k1', null);
    expect(inCurrentScope({ id: 'e', owner: null, operator_visible: true })).toBe(false);
    expect(inCurrentScope({ id: 'e', operator_visible: true })).toBe(false);
    expect(
      liveVerdict({ id: 'e', owner: null, operator_visible: true, session_id: '' }, 'new')
    ).toBe('out');
    setCurrentSessionId(null);
  });

  describe('liveVerdict (session view)', () => {
    test("a new artifact whose exec_id matches a held entry's is refetch, even when it would be in", async () => {
      await loadSessionView('k1', 'T1', [{ id: 'head', exec_id: 'run-1', owner: 'T1' }]);
      const result = { id: 'res', exec_id: 'run-1', owner: 'T1', operator_visible: true };
      expect(liveVerdict(result, 'new')).toBe('refetch');
      setCurrentSessionId(null);
    });

    test('an empty exec_id never matches', async () => {
      await loadSessionView('k1', 'T1', [{ id: 'head', exec_id: '', owner: 'T1' }]);
      expect(
        liveVerdict({ id: 'n', exec_id: '', owner: 'T1', operator_visible: true }, 'new')
      ).toBe('in');
      setCurrentSessionId(null);
    });

    test('an artifact the held transcript owns and the operator sees is in', async () => {
      await loadSessionView('k1', 'T1');
      expect(
        liveVerdict({ id: 'n', owner: 'T1', operator_visible: true, session_id: 'k1' }, 'new')
      ).toBe('in');
      expect(liveVerdict({ id: 'n', owner: 'T1', operator_visible: true }, 'updated')).toBe('in');
      setCurrentSessionId(null);
    });

    test('a new artifact carrying the current key while the held transcript is null or another is refetch', async () => {
      await loadSessionView('k1', null);
      expect(
        liveVerdict({ id: 'n', session_id: 'k1', owner: 'T9', operator_visible: true }, 'new')
      ).toBe('refetch');
      await loadSessionView('k1', 'T1');
      expect(
        liveVerdict({ id: 'n', session_id: 'k1', owner: 'T2', operator_visible: true }, 'new')
      ).toBe('refetch');
      setCurrentSessionId(null);
    });

    test('an updated frame never refetches', async () => {
      await loadSessionView('k1', null, [{ id: 'head', exec_id: 'run-1' }]);
      expect(
        liveVerdict({ id: 'n', session_id: 'k1', exec_id: 'run-1', owner: 'T9' }, 'updated')
      ).toBe('out');
      setCurrentSessionId(null);
    });

    test('anything else is out', async () => {
      await loadSessionView('k1', 'T1');
      expect(
        liveVerdict({ id: 'n', session_id: 'k2', owner: 'T2', operator_visible: true }, 'new')
      ).toBe('out');
      expect(liveVerdict({ id: 'n', owner: 'T1', operator_visible: false }, 'new')).toBe('out');
      setCurrentSessionId(null);
    });

    test('outside the session view every frame is in', async () => {
      await loadFirstPage({ artifacts: [{ id: 'head', exec_id: 'run-1' }] });
      expect(liveVerdict({ id: 'n', exec_id: 'run-1', owner: 'T2' }, 'new')).toBe('in');
    });
  });

  test('followsFocus: the session view follows its held transcript only; all-sessions follows every focus', async () => {
    await loadSessionView('k1', 'T1');
    expect(followsFocus('T1')).toBe(true);
    expect(followsFocus('T2')).toBe(false);
    expect(followsFocus(null)).toBe(false);
    await loadSessionView('k1', null);
    expect(followsFocus(null)).toBe(false);
    expect(followsFocus('T1')).toBe(false);
    setShowAllSessions(true);
    expect(followsFocus(null)).toBe(true);
    expect(followsFocus('T2')).toBe(true);
    setShowAllSessions(false);
    setCurrentSessionId(null);
    expect(followsFocus('T2')).toBe(true);
  });

  test('addArtifact returns false while a search is active and leaves the list alone', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(okResponse({ artifacts: [{ id: 'a1' }], total: 1 }))
    );
    await fetchArtifacts({ search: 'beam' });

    expect(addArtifact({ id: 'new' })).toBe(false);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1']);
  });

  test('addArtifact returns true without adding an entry another transcript owns', async () => {
    await loadSessionView('s1', 'T1', [{ id: 'a1', owner: 'T1' }]);

    expect(
      addArtifact({ id: 'foreign', session_id: 's2', owner: 'T2', operator_visible: true })
    ).toBe(true);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1']);
    expect(addArtifact({ id: 'mine', owner: 'T1', operator_visible: true })).toBe(true);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'mine']);
    setCurrentSessionId(null);
  });

  test('fetchMoreArtifacts in the session view posts the held cursor in the same body', async () => {
    setCurrentSessionId('s1');
    setShowAllSessions(false);
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          okResponse({ artifacts: [{ id: 'a1' }], next_cursor: 'c1', transcript: 'T1' })
        )
    );
    await fetchArtifacts({ search: 'orbit' });
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        okResponse({ artifacts: [{ id: 'a2' }], next_cursor: null, transcript: 'T1' })
      );
    vi.stubGlobal('fetch', fetchMock);

    await fetchMoreArtifacts();
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/artifacts/session-list');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({
      key: 's1',
      search: 'orbit',
      cursor: 'c1',
      runs: 'grouped',
    });
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2']);
    expect(getHeldTranscript()).toBe('T1');
    setCurrentSessionId(null);
  });

  test('addArtifact adds an entry and refuses a duplicate', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 1 });

    expect(addArtifact({ id: 'a2' })).toBe(true);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2']);

    expect(addArtifact({ id: 'a2' })).toBe(true);
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2']);
  });

  test('removeArtifact drops the entry and tolerates an id the list does not hold', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }, { id: 'a2' }], total: 5 });

    removeArtifact('a1');
    expect(getArtifacts().map((a) => a.id)).toEqual(['a2']);

    removeArtifact('missing');
    expect(getArtifacts().map((a) => a.id)).toEqual(['a2']);
  });
});

describe('picks', () => {
  afterEach(() => {
    clearPicked();
    setPickAnchor(null);
  });

  test('getPickedIds returns a fresh copy, so a caller cannot change the picks', () => {
    setPicked(['a', 'b']);
    const copy = getPickedIds();
    expect([...copy]).toEqual(['a', 'b']);
    copy.add('c');
    expect([...getPickedIds()]).toEqual(['a', 'b']);
    expect(getPickedIds()).not.toBe(copy);
  });

  test('setPicked replaces the picks', () => {
    setPicked(['a', 'b']);
    setPicked(['c']);
    expect([...getPickedIds()]).toEqual(['c']);
  });

  test('togglePicked adds a row that is not picked and removes one that is', () => {
    togglePicked('a');
    togglePicked('b');
    expect([...getPickedIds()]).toEqual(['a', 'b']);
    togglePicked('a');
    expect([...getPickedIds()]).toEqual(['b']);
  });

  test('clearPicked empties the picks', () => {
    setPicked(['a', 'b']);
    clearPicked();
    expect(getPickedIds().size).toBe(0);
  });

  test('getPickAnchor/setPickAnchor round-trip, including null', () => {
    expect(getPickAnchor()).toBeNull();
    setPickAnchor('a');
    expect(getPickAnchor()).toBe('a');
    setPickAnchor(null);
    expect(getPickAnchor()).toBeNull();
  });

  test('removeArtifact also drops the id from the picks', () => {
    setArtifacts([{ id: 'a' }, { id: 'b' }]);
    setPicked(['a', 'b']);
    removeArtifact('a');
    expect([...getPickedIds()]).toEqual(['b']);
  });

  test('a first-page fetch clears the picks and the anchor', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ artifacts: [{ id: 'a' }, { id: 'b' }] }),
      })
    );
    setPicked(['a', 'b']);
    setPickAnchor('a');

    await fetchArtifacts({ search: 'beam' });

    expect(getPickedIds().size).toBe(0);
    expect(getPickAnchor()).toBeNull();
  });
});
