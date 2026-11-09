/**
 * Unit tests for the Artifact Gallery state/fetch layer (state.js).
 *
 * Pure-logic/DOM guard, happy-dom environment (configured globally), `fetch`
 * mocked via vi.stubGlobal — mirrors tests/interfaces/web_terminal/scaffold/data.test.mjs:
 *   npx vitest run tests/interfaces/artifacts/state.test.mjs
 *
 * Covers accessor correctness (every get/set pair round-trips), fileUrl,
 * the error banner, fetchArtifacts (success + both failure paths), and the session
 * scope.
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
  getCurrentSessionId,
  setCurrentSessionId,
  fileUrl,
  showErrorBanner,
  hideErrorBanner,
  fetchArtifacts,
  inCurrentScope,
  getHeldTranscript,
  addArtifact,
  removeArtifact,
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

  test('getCurrentSessionId/setCurrentSessionId round-trip, including null', () => {
    setCurrentSessionId('session-abc');
    expect(getCurrentSessionId()).toBe('session-abc');
    setCurrentSessionId(null);
    expect(getCurrentSessionId()).toBeNull();
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
  test('on success: updates artifacts, hides the error banner,', async () => {
    setCurrentSessionId(null);
    showErrorBanner('stale error');
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ artifacts: [{ id: 'a1' }] }),
      })
    );

    await fetchArtifacts();

    expect(getArtifacts()).toEqual([{ id: 'a1' }]);
    expect(byId('error-banner').style.display).toBe('none');
  });

  test('in the session view, posts the gallery body to session-list and holds the transcript', async () => {
    setCurrentSessionId('sess-42');
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
    setCurrentSessionId(null);
  });

  test('a key the ledger has no row for holds a null transcript', async () => {
    setCurrentSessionId('sess-unseen');
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
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ ok: true, json: () => Promise.resolve({ artifacts: [{ id: 'w1' }] }) });
    vi.stubGlobal('fetch', fetchMock);

    await fetchArtifacts();
    expect(fetchMock).toHaveBeenCalledWith('/api/artifacts');
    expect(getHeldTranscript()).toBeNull();
    expect(getArtifacts()).toEqual([{ id: 'w1' }]);
  });

  test('on a non-OK response: shows the error banner', async () => {
    setCurrentSessionId(null);
    setArtifacts([{ id: 'kept' }]);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        text: () => Promise.resolve('boom'),
      })
    );

    await fetchArtifacts();

    expect(getArtifacts()).toEqual([{ id: 'kept' }]);
    expect(byId('error-banner').textContent).toContain('API error (500)');
  });

  test('on a network failure: shows the error banner with the error message', async () => {
    setCurrentSessionId(null);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));

    await fetchArtifacts();

    expect(byId('error-banner').textContent).toBe('Failed to fetch artifacts: network down');
  });

  test('resolves to nothing', async () => {
    setCurrentSessionId(null);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ artifacts: [] }) })
    );
    await expect(fetchArtifacts()).resolves.toBeUndefined();
  });
});

describe('the session scope', () => {
  /** @param {any} body */
  function okResponse(body) {
    return { ok: true, json: () => Promise.resolve(body) };
  }

  /** @param {any} firstPage */
  async function loadFirstPage(firstPage) {
    setCurrentSessionId(null);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(okResponse(firstPage)));
    await fetchArtifacts();
  }

  /**
   * Load a session-view first page for key `key` whose answer names `transcript`.
   * @param {string} key @param {string|null} transcript @param {any[]} [list]
   */
  async function loadSessionView(key, transcript, list = []) {
    setCurrentSessionId(key);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(okResponse({ artifacts: list, transcript, ledger: 'ok' }))
    );
    await fetchArtifacts();
  }

  test('inCurrentScope admits what the held transcript owns, and everything outside a session on', async () => {
    await loadSessionView('k1', 'T1');
    expect(inCurrentScope({ id: 'x', owner: 'T1', operator_visible: true })).toBe(true);
    expect(inCurrentScope({ id: 'x', owner: 'T1', operator_visible: false })).toBe(false);
    expect(inCurrentScope({ id: 'x', owner: 'T2', operator_visible: true })).toBe(false);
    expect(inCurrentScope({ id: 'x', owner: null, operator_visible: true })).toBe(false);
    // The key alone does not admit: the transcript that owns it does.
    expect(inCurrentScope({ id: 'x', session_id: 'k1', owner: 'T2', operator_visible: true })).toBe(
      false
    );
    setCurrentSessionId(null);
    expect(inCurrentScope({ id: 'x', owner: 'T2' })).toBe(true);
  });

  test('a held null transcript admits nothing, not even an unowned artifact', async () => {
    await loadSessionView('k1', null);
    expect(inCurrentScope({ id: 'e', owner: null, operator_visible: true })).toBe(false);
    expect(inCurrentScope({ id: 'e', operator_visible: true })).toBe(false);
    setCurrentSessionId(null);
  });

  test('addArtifact does not hold an entry another transcript owns', async () => {
    await loadSessionView('s1', 'T1', [{ id: 'a1', owner: 'T1' }]);

    addArtifact({ id: 'foreign', session_id: 's2', owner: 'T2', operator_visible: true });
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1']);
    addArtifact({ id: 'mine', owner: 'T1', operator_visible: true });
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'mine']);
    setCurrentSessionId(null);
  });

  test('addArtifact adds an entry and refuses a duplicate', async () => {
    await loadFirstPage({ artifacts: [{ id: 'a1' }], total: 1 });

    addArtifact({ id: 'a2' });
    expect(getArtifacts().map((a) => a.id)).toEqual(['a1', 'a2']);

    addArtifact({ id: 'a2' });
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
