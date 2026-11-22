// @ts-check
/**
 * The KNOWLEDGE panel's search list shows the answer to the newest query
 * (app.js's search token):
 *   npx vitest run tests/interfaces/okf_panel/app-search-race.test.mjs
 *
 * Each search awaits its answer, so the answers to two queries may arrive in
 * either order; the one asked for last is the one the list shows, and a query
 * since cleared shows nothing. The fetch stub honours the abort signal, so a
 * superseded search that the panel aborts rejects the way a browser's fetch
 * does.
 *
 * app.js boots on evaluation, so the DOM, `marked` and `fetch` are in place
 * before the one dynamic import below.
 */

import { beforeAll, describe, expect, test, vi } from 'vitest';

import { mountShell } from './page-shell.mjs';

/**
 * @param {unknown} body
 */
function jsonResponse(body) {
  return { ok: true, status: 200, json: async () => body };
}

/** Answers held until a test releases them, by url. @type {Map<string, () => void>} */
const held = new Map();
/** Urls to hold the next time they are fetched. @type {Set<string>} */
const toHold = new Set();

/**
 * @param {string} q
 */
function searchUrl(q) {
  return '/api/search?q=' + encodeURIComponent(q);
}

/**
 * @param {string} url
 */
function answer(url) {
  if (url === '/api/concepts') return jsonResponse({ groups: [] });
  if (url === '/api/bundle_health') return jsonResponse({ ok: true, counts: {}, total: 0 });
  if (url === '/api/structure') return jsonResponse({ markdown: '# Structure' });
  if (url.startsWith('/api/search?q=')) {
    const q = decodeURIComponent(url.slice('/api/search?q='.length));
    return jsonResponse({ results: [{ id: 'hit/' + q, title: 'hit ' + q }] });
  }
  throw new Error(`unstubbed fetch: ${url}`);
}

/**
 * @param {string} url
 * @param {{signal?: AbortSignal}} [init]
 */
async function fakeFetch(url, init) {
  if (toHold.has(url)) {
    toHold.delete(url);
    await new Promise((resolve) => held.set(url, () => resolve(undefined)));
  }
  if (init?.signal?.aborted) throw new DOMException('aborted', 'AbortError');
  return answer(url);
}

async function flush() {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * @param {string} url
 */
async function release(url) {
  const go = held.get(url);
  if (!go) throw new Error(`nothing held for ${url}`);
  held.delete(url);
  go();
  await flush();
}

function input() {
  return /** @type {HTMLInputElement} */ (document.getElementById('search-input'));
}

/**
 * Type `value` into the search field and press Enter, which searches at once.
 * @param {string} value
 */
async function search(value) {
  input().value = value;
  input().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
  await flush();
}

function results() {
  return /** @type {HTMLElement} */ (document.getElementById('search-results'));
}

function shownHits() {
  return [...results().querySelectorAll('.result-link')].map(
    (a) => /** @type {HTMLElement} */ (a).dataset.conceptId
  );
}

beforeAll(async () => {
  mountShell();
  vi.stubGlobal('marked', { parse: (/** @type {string} */ md) => md });
  vi.stubGlobal('fetch', vi.fn(fakeFetch));
  await import('../../../src/osprey/interfaces/okf_panel/static/js/app.js');
  await flush();
});

describe('okf panel search', () => {
  test("of two queries, the list shows the newer one's hits when the older answers last", async () => {
    toHold.add(searchUrl('a'));
    await search('a');
    await search('ab');
    expect(shownHits()).toEqual(['hit/ab']);

    await release(searchUrl('a'));

    expect(shownHits()).toEqual(['hit/ab']);
    expect(results().textContent).not.toContain('Search failed.');
  });

  test('a query cleared while its answer is out leaves the list empty', async () => {
    toHold.add(searchUrl('a'));
    await search('a');
    await search('');
    expect(results().hidden).toBe(true);

    await release(searchUrl('a'));

    expect(results().hidden).toBe(true);
    expect(shownHits()).toEqual([]);
  });
});
