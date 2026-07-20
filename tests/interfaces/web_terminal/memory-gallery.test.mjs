// @ts-check
/**
 * Unit tests for the Web Terminal Memory Gallery's write actions
 * (memory-gallery.js's `saveFile`/`deleteCurrentFile`/`promptCreateFile`,
 * driven through the real DOM via `initMemoryGallery()`):
 *   npx vitest run tests/interfaces/web_terminal/memory-gallery.test.mjs
 *
 * Covers the raw PUT/DELETE/POST `fetch()` calls being prefix-aware via
 * `window.__OSPREY_PREFIX__` (multi-user deployments) -- `fetchJSON`
 * (api.js) already prefixes the GET list/detail loads, but these write
 * actions are raw `fetch()` calls with request options `fetchJSON` doesn't
 * support, so this module applies the shared `withPrefix` helper (imported
 * from api.js) to their paths directly.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { initMemoryGallery } from '../../../src/osprey/interfaces/web_terminal/static/js/memory-gallery.js';

const FILE = {
  filename: 'topic-a.md',
  is_primary: false,
  line_count: 10,
  size: 200,
};

/** Mount the minimal DOM `initMemoryGallery()` needs (drawer + tab panel). */
function mountFixture() {
  document.body.innerHTML = `
    <div id="settings-drawer"></div>
    <div id="tab-memory"></div>
  `;
  const drawer = /** @type {any} */ (document.getElementById('settings-drawer'));
  drawer.registerUnsavedGuard = () => {};
}

/**
 * Route the stubbed `fetch` by URL substring: the list/detail GETs (via
 * fetchJSON) resolve with fixed payloads; anything else falls through to
 * `writeResponse` so a single stub covers both the initial load and the
 * write action under test.
 * @param {any} writeResponse
 */
function stubFetch(writeResponse) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url) => {
      if (typeof url === 'string' && url.endsWith('/api/claude-memory')) {
        return { ok: true, json: async () => ({ files: [FILE] }) };
      }
      if (typeof url === 'string' && url.includes('/api/claude-memory/')) {
        return { ok: true, json: async () => ({ content: 'hello' }) };
      }
      return writeResponse;
    })
  );
}

/** Flush the microtask queue so awaited fetch chains settle. */
function flush() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

beforeEach(() => {
  mountFixture();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  delete window.__OSPREY_PREFIX__;
});

/** Init, load the gallery, and open the fixture file's detail view. */
async function openFixtureDetail() {
  initMemoryGallery();
  document.getElementById('tab-memory')?.dispatchEvent(new Event('drawer:tab-activate'));
  await flush();
  /** @type {HTMLElement} */ (document.querySelector('.memory-file-card')).click();
  await flush();
}

describe('deleteCurrentFile', () => {
  test('prepends window.__OSPREY_PREFIX__ to the DELETE (multi-user deployments)', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    stubFetch({ ok: true });
    vi.stubGlobal('confirm', vi.fn(() => true));

    await openFixtureDetail();
    /** @type {HTMLElement} */ (document.querySelector('.memory-delete-btn')).click();
    await flush();

    const fetchMock = /** @type {import('vitest').Mock} */ (fetch);
    const deleteCall = fetchMock.mock.calls.find(([, init]) => init?.method === 'DELETE');
    expect(deleteCall?.[0]).toBe(`/u/alice/api/claude-memory/${encodeURIComponent(FILE.filename)}`);
  });
});

describe('saveFile', () => {
  test('prepends window.__OSPREY_PREFIX__ to the PUT (multi-user deployments)', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    stubFetch({ ok: true, json: async () => FILE });

    await openFixtureDetail();
    const textarea = /** @type {HTMLTextAreaElement} */ (
      document.querySelector('.memory-edit-textarea')
    );
    textarea.value = 'edited content';
    // The save button starts disabled (editDirty === false); an 'input'
    // event flips editDirty and re-renders the actions bar (a new node),
    // so the save button must be re-queried afterward.
    textarea.dispatchEvent(new Event('input'));
    /** @type {HTMLElement} */ (document.querySelector('.memory-save-btn')).click();
    await flush();

    const fetchMock = /** @type {import('vitest').Mock} */ (fetch);
    const putCall = fetchMock.mock.calls.find(([, init]) => init?.method === 'PUT');
    expect(putCall?.[0]).toBe(`/u/alice/api/claude-memory/${encodeURIComponent(FILE.filename)}`);
  });
});

describe('promptCreateFile', () => {
  test('prepends window.__OSPREY_PREFIX__ to the POST (multi-user deployments)', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    stubFetch({ ok: true, json: async () => ({ filename: 'new-topic.md' }) });
    vi.stubGlobal('prompt', vi.fn(() => 'new-topic.md'));

    initMemoryGallery();
    document.getElementById('tab-memory')?.dispatchEvent(new Event('drawer:tab-activate'));
    await flush();
    /** @type {HTMLElement} */ (document.querySelector('.memory-new-btn')).click();
    await flush();

    const fetchMock = /** @type {import('vitest').Mock} */ (fetch);
    const postCall = fetchMock.mock.calls.find(([, init]) => init?.method === 'POST');
    expect(postCall?.[0]).toBe('/u/alice/api/claude-memory');
  });
});
