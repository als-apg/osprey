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
 * paths), and getFilteredArtifacts' filter/search/sort combinations over
 * fixture artifacts.
 *
 * NOTE: state.js exports module-singleton state (there's only ever one
 * gallery per page, so a single shared instance is by design). Tests that
 * touch shared state call the relevant setters first, so execution order
 * between tests doesn't matter.
 */

import { test, expect, vi, describe, afterEach } from 'vitest';

import {
  getArtifacts,
  setArtifacts,
  getSelectedArtifact,
  setSelectedArtifact,
  getFocusedArtifact,
  setFocusedArtifact,
  getActiveFilter,
  setActiveFilter,
  getCurrentSessionId,
  setCurrentSessionId,
  getShowAllSessions,
  setShowAllSessions,
  fileUrl,
  showErrorBanner,
  hideErrorBanner,
  fetchArtifacts,
  fetchFocus,
  getFilteredArtifacts,
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

  test('getActiveFilter/setActiveFilter round-trip', () => {
    setActiveFilter('pinned');
    expect(getActiveFilter()).toBe('pinned');
    setActiveFilter('plot_html');
    expect(getActiveFilter()).toBe('plot_html');
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
    expect(banner.textContent).toBe('something broke');
    expect(banner.style.display).toBe('block');
  });

  test('a second showErrorBanner call reuses the existing element', () => {
    showErrorBanner('first message');
    const first = document.getElementById('error-banner');
    showErrorBanner('second message');
    const second = document.getElementById('error-banner');
    expect(second).toBe(first);
    expect(second.textContent).toBe('second message');
  });

  test('hideErrorBanner hides an existing banner without throwing when none exists', () => {
    showErrorBanner('to be hidden');
    hideErrorBanner();
    expect(document.getElementById('error-banner').style.display).toBe('none');

    document.getElementById('error-banner').remove();
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
    expect(document.getElementById('error-banner').style.display).toBe('none');
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
    expect(document.getElementById('error-banner').textContent).toContain('API error (500)');
  });

  test('on a network failure: shows the error banner with the error message and fires onHealthChange(false)', async () => {
    setCurrentSessionId(null);
    setShowAllSessions(false);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));

    const onHealthChange = vi.fn();
    await fetchArtifacts({ onHealthChange });

    expect(onHealthChange).toHaveBeenCalledWith(false);
    expect(document.getElementById('error-banner').textContent).toBe('Failed to fetch artifacts: network down');
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

  test('activeFilter "all" with no search returns everything, pinned first then newest-first', () => {
    setArtifacts(makeFixtures());
    setActiveFilter('all');

    const result = getFilteredArtifacts('');
    expect(result.map((a) => a.id)).toEqual(['2', '4', '3', '1']);
  });

  test('activeFilter "pinned" returns only pinned artifacts', () => {
    setArtifacts(makeFixtures());
    setActiveFilter('pinned');

    const result = getFilteredArtifacts('');
    expect(result.map((a) => a.id).sort()).toEqual(['2', '4']);
  });

  test('a type/category filter matches on category OR artifact_type', () => {
    setArtifacts(makeFixtures());
    setActiveFilter('visualization');

    const result = getFilteredArtifacts('');
    expect(result.map((a) => a.id).sort()).toEqual(['1', '3']);
  });

  test('search matches title, filename, description, or artifact_type (case-insensitive)', () => {
    setArtifacts(makeFixtures());
    setActiveFilter('all');

    expect(getFilteredArtifacts('beam').map((a) => a.id)).toEqual(['1']);
    expect(getFilteredArtifacts('channels.json').map((a) => a.id)).toEqual(['2']);
    expect(getFilteredArtifacts('magnet strengths').map((a) => a.id)).toEqual(['3']);
    expect(getFilteredArtifacts('markdown').map((a) => a.id)).toEqual(['4']);
  });

  test('filter and search compose: search narrows within the active filter', () => {
    setArtifacts(makeFixtures());
    setActiveFilter('visualization');

    expect(getFilteredArtifacts('lattice').map((a) => a.id)).toEqual(['3']);
    expect(getFilteredArtifacts('report').map((a) => a.id)).toEqual([]);
  });

  test('an empty search query (default) is treated as no search filter', () => {
    setArtifacts(makeFixtures());
    setActiveFilter('all');
    expect(getFilteredArtifacts().length).toBe(4);
  });

  test('does not mutate the underlying artifacts array', () => {
    const fixtures = makeFixtures();
    setArtifacts(fixtures);
    setActiveFilter('all');

    getFilteredArtifacts('');
    expect(getArtifacts()).toBe(fixtures);
    expect(fixtures.map((a) => a.id)).toEqual(['1', '2', '3', '4']);
  });
});
