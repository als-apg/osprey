/**
 * Unit tests for the Scaffold Gallery data layer (scaffold/data.js).
 *
 * Pure-logic guard, happy-dom environment (configured globally), `fetch`/
 * `confirm` mocked via vi.stubGlobal — mirrors the pattern in
 * tests/interfaces/lattice_dashboard/net.test.mjs:
 *   npx vitest run tests/interfaces/web_terminal/scaffold-data.test.mjs
 *
 * Covers the shared fetch-cache single-flight behavior, resetFetchCache
 * invalidation, loadArtifacts' filter/map/summary computation and its two
 * error paths (a failed /api/scaffold fetch propagates; a failed
 * /api/scaffold/untracked fetch is swallowed), the register/delete
 * untracked-file actions, and createScaffoldDataActions' callback wiring
 * (the factory that mirrors lattice_dashboard/net.js's createNetClient
 * pattern — pure functions above, bound to named-after-effect callbacks).
 *
 * NOTE: imported by RELATIVE path — this module lives under web_terminal,
 * not design-system, so the `/design-system/js/*` alias does not apply.
 */

import { test, expect, vi, describe, beforeEach, afterEach } from 'vitest';

import {
  fetchArtifactsShared,
  resetFetchCache,
  loadArtifacts,
  registerUntrackedFile,
  deleteUntrackedFile,
  createScaffoldDataActions,
} from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/data.js';

/**
 * @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/scaffold/data.js').ArtifactFilterState} ArtifactFilterState
 */

/**
 * A loaded/untracked artifact as shaped by these tests' fixtures. The
 * production {@link ArtifactFilterState.categoryFilter}/data-layer types are
 * intentionally loose (`object[]`) since the module is domain-agnostic; the
 * tests fix a concrete shape to exercise property access.
 * @typedef {object} TestArtifact
 * @property {string} name
 * @property {string} category
 * @property {string} [status]
 * @property {string} [displayCategory]
 */

/**
 * @typedef {object} TestUntrackedFile
 * @property {string} canonical_name
 * @property {string} category
 * @property {string} [output_path]
 */

/** @param {Partial<ArtifactFilterState>} [overrides]
 * @returns {ArtifactFilterState}
 */
function makeState(overrides = {}) {
  return {
    categoryFilter: () => true,
    categoryOverrides: {},
    categoryRemaps: {},
    ...overrides,
  };
}

beforeEach(() => {
  // The fetch cache is module-level singleton state; start each test clean.
  resetFetchCache();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('fetchArtifactsShared / resetFetchCache', () => {
  test('two concurrent callers share a single underlying fetch (single-flight)', async () => {
    let fetchCalls = 0;
    vi.stubGlobal('fetch', vi.fn(() => {
      fetchCalls++;
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [{ name: 'a' }] }) });
    }));

    const [r1, r2] = await Promise.all([fetchArtifactsShared(), fetchArtifactsShared()]);

    expect(fetchCalls).toBe(1);
    expect(r1).toBe(r2);
  });

  test('resetFetchCache forces a fresh fetch on the next call', async () => {
    let fetchCalls = 0;
    vi.stubGlobal('fetch', vi.fn(() => {
      fetchCalls++;
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
    }));

    await fetchArtifactsShared();
    expect(fetchCalls).toBe(1);

    resetFetchCache();

    await fetchArtifactsShared();
    expect(fetchCalls).toBe(2);
  });

  test('without a reset, a second call reuses the cached promise (no new fetch)', async () => {
    let fetchCalls = 0;
    vi.stubGlobal('fetch', vi.fn(() => {
      fetchCalls++;
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
    }));

    await fetchArtifactsShared();
    await fetchArtifactsShared();
    expect(fetchCalls).toBe(1);
  });
});

describe('loadArtifacts', () => {
  test('filters to the domain, applies category overrides/remaps, and computes the summary', async () => {
    vi.stubGlobal('fetch', vi.fn((url) => {
      if (url === '/api/scaffold') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            artifacts: [
              { name: 'agent-a', category: 'agents', status: 'framework' },
              { name: 'agent-b', category: 'agents', status: 'user-owned' },
              { name: 'hook-a', category: 'hooks', status: 'framework' },
            ],
          }),
        });
      }
      if (url === '/api/scaffold/untracked') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ untracked: [] }) });
      }
      return Promise.reject(new Error(`unexpected fetch ${url}`));
    }));

    const state = makeState({ categoryFilter: (a) => a.category === 'agents' });
    const result = await loadArtifacts(state);
    const artifacts = /** @type {TestArtifact[]} */ (result.artifacts);

    expect(result.artifacts).toHaveLength(2);
    expect(artifacts.map((a) => a.name)).toEqual(['agent-a', 'agent-b']);
    expect(artifacts[0].displayCategory).toBe('agents');
    expect(result.summary).toEqual({ total: 2, framework: 1, userOwned: 1 });
  });

  test('a per-name categoryOverride wins over a per-category categoryRemap', async () => {
    vi.stubGlobal('fetch', vi.fn((url) => {
      if (url === '/api/scaffold') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            artifacts: [{ name: 'claude-md', category: 'config', status: 'framework' }],
          }),
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ untracked: [] }) });
    }));

    const state = makeState({
      categoryOverrides: { 'claude-md': 'system instructions' },
      categoryRemaps: { config: 'settings' },
    });
    const result = await loadArtifacts(state);
    const artifacts = /** @type {TestArtifact[]} */ (result.artifacts);

    expect(artifacts[0].displayCategory).toBe('system instructions');
  });

  test('untracked files are matched via categoryFilter on either the raw or remapped category', async () => {
    vi.stubGlobal('fetch', vi.fn((url) => {
      if (url === '/api/scaffold') {
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
      }
      if (url === '/api/scaffold/untracked') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            untracked: [
              { canonical_name: 'stray-hook', category: 'hooks', output_path: '.claude/hooks/stray.py' },
              { canonical_name: 'stray-agent', category: 'agents', output_path: '.claude/agents/stray.md' },
            ],
          }),
        });
      }
      return Promise.reject(new Error(`unexpected fetch ${url}`));
    }));

    const state = makeState({ categoryFilter: (a) => a.category === 'hooks' });
    const result = await loadArtifacts(state);
    const untrackedFiles = /** @type {TestUntrackedFile[]} */ (result.untrackedFiles);

    expect(result.untrackedFiles).toHaveLength(1);
    expect(untrackedFiles[0].canonical_name).toBe('stray-hook');
  });

  test('skipCache invalidates the shared cache before fetching', async () => {
    let scaffoldCalls = 0;
    vi.stubGlobal('fetch', vi.fn((url) => {
      if (url === '/api/scaffold') {
        scaffoldCalls++;
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ artifacts: [] }) });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ untracked: [] }) });
    }));

    await fetchArtifactsShared();
    expect(scaffoldCalls).toBe(1);

    await loadArtifacts(makeState(), { skipCache: true });
    expect(scaffoldCalls).toBe(2);

    await loadArtifacts(makeState());
    expect(scaffoldCalls).toBe(2); // no skipCache this time: cached promise reused
  });

  test('a failed /api/scaffold fetch propagates as a rejection (action error path)', async () => {
    vi.stubGlobal('fetch', vi.fn((url) => {
      if (url === '/api/scaffold') return Promise.reject(new TypeError('network down'));
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ untracked: [] }) });
    }));

    await expect(loadArtifacts(makeState())).rejects.toThrow('network down');
  });

  test('a failed /api/scaffold/untracked fetch is swallowed: load still succeeds with an empty list', async () => {
    vi.stubGlobal('fetch', vi.fn((url) => {
      if (url === '/api/scaffold') {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ artifacts: [{ name: 'a', category: 'agents', status: 'framework' }] }),
        });
      }
      return Promise.reject(new TypeError('network down'));
    }));

    const result = await loadArtifacts(makeState());
    expect(result.untrackedFiles).toEqual([]);
    expect(result.artifacts).toHaveLength(1);
  });
});

describe('registerUntrackedFile', () => {
  test('resolves without throwing on a 200 response, POSTing the canonical name', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal('fetch', fetchMock);

    await expect(registerUntrackedFile('my-hook')).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith('/api/scaffold/untracked/register', expect.objectContaining({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'my-hook' }),
    }));
  });

  test('throws with the API-provided detail message on a non-OK response (action error path)', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ detail: 'name already tracked' }),
    }));

    await expect(registerUntrackedFile('dup')).rejects.toThrow('name already tracked');
  });

  test('falls back to a generic HTTP-status message when the error body has no detail', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: () => Promise.reject(new Error('not json')),
    }));

    await expect(registerUntrackedFile('x')).rejects.toThrow('Register failed (HTTP 500)');
  });
});

describe('deleteUntrackedFile', () => {
  test('does not call fetch and resolves false when the operator declines the confirmation', async () => {
    vi.stubGlobal('confirm', vi.fn(() => false));
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    await expect(deleteUntrackedFile('x')).resolves.toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test('deletes and resolves true when confirmed and the response is OK', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true));
    const fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal('fetch', fetchMock);

    await expect(deleteUntrackedFile('my file')).resolves.toBe(true);
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/scaffold/untracked/${encodeURIComponent('my file')}`,
      expect.objectContaining({ method: 'DELETE' })
    );
  });

  test('throws with the API-provided detail message on a non-OK response (action error path)', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      json: () => Promise.resolve({ detail: 'file not found' }),
    }));

    await expect(deleteUntrackedFile('gone')).rejects.toThrow('file not found');
  });
});

describe('createScaffoldDataActions (callback-bound actions)', () => {
  /** @returns {{onLoadStart: import('vitest').Mock, onLoaded: import('vitest').Mock, onLoadError: import('vitest').Mock}} */
  function makeCallbacks() {
    return { onLoadStart: vi.fn(), onLoaded: vi.fn(), onLoadError: vi.fn() };
  }

  /**
   * Routes fetch by exact URL; falls through to a 404-ish rejection for anything unmapped.
   * @param {Record<string, {ok: boolean, status?: number, json?: () => Promise<unknown>}>} routes
   */
  function stubFetchRoutes(routes) {
    vi.stubGlobal('fetch', vi.fn((url) => {
      const route = routes[url];
      if (!route) return Promise.reject(new Error(`unexpected fetch ${url}`));
      return Promise.resolve(route);
    }));
  }

  test('load() fires onLoadStart synchronously, then onLoaded with the fresh data on success', () => {
    stubFetchRoutes({
      '/api/scaffold': { ok: true, json: () => Promise.resolve({ artifacts: [{ name: 'a', category: 'agents', status: 'framework' }] }) },
      '/api/scaffold/untracked': { ok: true, json: () => Promise.resolve({ untracked: [] }) },
    });

    const callbacks = makeCallbacks();
    const actions = createScaffoldDataActions(makeState({ categoryFilter: (a) => a.category === 'agents' }), callbacks);

    const loadPromise = actions.load();
    // onLoadStart fires synchronously, before the fetch resolves.
    expect(callbacks.onLoadStart).toHaveBeenCalledTimes(1);
    expect(callbacks.onLoaded).not.toHaveBeenCalled();

    return loadPromise.then(() => {
      expect(callbacks.onLoaded).toHaveBeenCalledTimes(1);
      expect(callbacks.onLoaded.mock.calls[0][0].summary).toEqual({ total: 1, framework: 1, userOwned: 0 });
      expect(callbacks.onLoadError).not.toHaveBeenCalled();
    });
  });

  test('load() fires onLoadError with a "Failed to load prompts" prefix when the fetch rejects', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));

    const callbacks = makeCallbacks();
    const actions = createScaffoldDataActions(makeState(), callbacks);

    await actions.load();

    expect(callbacks.onLoadStart).toHaveBeenCalledTimes(1);
    expect(callbacks.onLoaded).not.toHaveBeenCalled();
    expect(callbacks.onLoadError).toHaveBeenCalledWith('Failed to load prompts: network down');
  });

  test('reloadFull() fires onLoaded WITHOUT onLoadStart (matches the original: no loading-spinner toggle on refresh)', async () => {
    stubFetchRoutes({
      '/api/scaffold': { ok: true, json: () => Promise.resolve({ artifacts: [] }) },
      '/api/scaffold/untracked': { ok: true, json: () => Promise.resolve({ untracked: [] }) },
    });

    const callbacks = makeCallbacks();
    const actions = createScaffoldDataActions(makeState(), callbacks);

    await actions.reloadFull();

    expect(callbacks.onLoadStart).not.toHaveBeenCalled();
    expect(callbacks.onLoaded).toHaveBeenCalledTimes(1);
  });

  test('registerUntracked() registers, reloads, and fires onLoaded on success', async () => {
    stubFetchRoutes({
      '/api/scaffold/untracked/register': { ok: true },
      '/api/scaffold': { ok: true, json: () => Promise.resolve({ artifacts: [] }) },
      '/api/scaffold/untracked': { ok: true, json: () => Promise.resolve({ untracked: [] }) },
    });

    const callbacks = makeCallbacks();
    const actions = createScaffoldDataActions(makeState(), callbacks);

    await actions.registerUntracked('my-hook');

    expect(callbacks.onLoaded).toHaveBeenCalledTimes(1);
    expect(callbacks.onLoadError).not.toHaveBeenCalled();
  });

  test('registerUntracked() fires onLoadError with a "Register failed" prefix when the register call fails', async () => {
    stubFetchRoutes({
      '/api/scaffold/untracked/register': {
        ok: false,
        status: 400,
        json: () => Promise.resolve({ detail: 'name already tracked' }),
      },
    });

    const callbacks = makeCallbacks();
    const actions = createScaffoldDataActions(makeState(), callbacks);

    await actions.registerUntracked('dup');

    expect(callbacks.onLoaded).not.toHaveBeenCalled();
    expect(callbacks.onLoadError).toHaveBeenCalledWith('Register failed: name already tracked');
  });

  test('deleteUntracked() neither reloads nor fires a callback when the operator declines the confirmation', async () => {
    vi.stubGlobal('confirm', vi.fn(() => false));
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const callbacks = makeCallbacks();
    const actions = createScaffoldDataActions(makeState(), callbacks);

    await actions.deleteUntracked('x');

    expect(fetchMock).not.toHaveBeenCalled();
    expect(callbacks.onLoaded).not.toHaveBeenCalled();
    expect(callbacks.onLoadError).not.toHaveBeenCalled();
  });

  test('deleteUntracked() deletes, reloads, and fires onLoaded when confirmed and the response is OK', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true));
    stubFetchRoutes({
      [`/api/scaffold/untracked/${encodeURIComponent('my file')}`]: { ok: true },
      '/api/scaffold': { ok: true, json: () => Promise.resolve({ artifacts: [] }) },
      '/api/scaffold/untracked': { ok: true, json: () => Promise.resolve({ untracked: [] }) },
    });

    const callbacks = makeCallbacks();
    const actions = createScaffoldDataActions(makeState(), callbacks);

    await actions.deleteUntracked('my file');

    expect(callbacks.onLoaded).toHaveBeenCalledTimes(1);
    expect(callbacks.onLoadError).not.toHaveBeenCalled();
  });

  test('deleteUntracked() fires onLoadError with a "Delete failed" prefix when the delete call fails', async () => {
    vi.stubGlobal('confirm', vi.fn(() => true));
    stubFetchRoutes({
      '/api/scaffold/untracked/gone': {
        ok: false,
        status: 404,
        json: () => Promise.resolve({ detail: 'file not found' }),
      },
    });

    const callbacks = makeCallbacks();
    const actions = createScaffoldDataActions(makeState(), callbacks);

    await actions.deleteUntracked('gone');

    expect(callbacks.onLoaded).not.toHaveBeenCalled();
    expect(callbacks.onLoadError).toHaveBeenCalledWith('Delete failed: file not found');
  });
});
