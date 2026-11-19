// @ts-check
/**
 * The KNOWLEDGE panel's reading pane shows the page it was asked for last
 * (app.js's reader load token):
 *   npx vitest run tests/interfaces/okf_panel/app-reader-race.test.mjs
 *
 * A panel that boots on no fragment loads the structure overview. A citation
 * that opens the panel on a concept while that load is in flight starts a
 * second load. Each load awaits its answer, so the answers may arrive in
 * either order; the one asked for last is the one the reader shows.
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
const toHold = new Set(['/api/structure']);

/**
 * @param {string} url
 */
function answer(url) {
  if (url === '/api/concepts') {
    return jsonResponse({
      groups: [
        {
          id: 'devices',
          label: 'devices',
          concepts: [
            { id: 'devices/bpm', title: 'BPM' },
            { id: 'devices/corrector', title: 'Corrector' },
          ],
        },
      ],
    });
  }
  if (url === '/api/bundle_health') return jsonResponse({ ok: true, counts: {}, total: 0 });
  if (url === '/api/structure') return jsonResponse({ markdown: '# Structure' });
  if (url.startsWith('/api/concept?id=')) {
    const id = decodeURIComponent(url.slice('/api/concept?id='.length));
    return jsonResponse({ id, frontmatter: { title: id }, body: '' });
  }
  throw new Error(`unstubbed fetch: ${url}`);
}

/**
 * @param {string} url
 */
async function fakeFetch(url) {
  if (toHold.has(url)) {
    toHold.delete(url);
    await new Promise((resolve) => held.set(url, () => resolve(undefined)));
  }
  return answer(url);
}

async function flush() {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * @param {string} hash
 */
async function changeHash(hash) {
  location.hash = hash;
  await flush();
  window.dispatchEvent(new HashChangeEvent('hashchange'));
  await flush();
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

function reader() {
  return /** @type {HTMLElement} */ (document.getElementById('reader-content'));
}

beforeAll(async () => {
  mountShell();
  vi.stubGlobal('marked', { parse: (/** @type {string} */ md) => md });
  vi.stubGlobal('fetch', vi.fn(fakeFetch));
  await import('../../../src/osprey/interfaces/okf_panel/static/js/app.js');
  await flush();
});

describe('okf panel reader loads', () => {
  test('a concept opened while the boot overview loads stays shown when the overview answers late', async () => {
    expect(held.has('/api/structure')).toBe(true);
    await changeHash('#devices/bpm');
    expect(reader().dataset.conceptId).toBe('devices/bpm');

    await release('/api/structure');

    expect(reader().dataset.conceptId).toBe('devices/bpm');
    expect(reader().querySelector('h1')?.textContent).toBe('devices/bpm');
  });

  test('of two concepts, the one asked for last is shown whatever answers first', async () => {
    const first = '/api/concept?id=' + encodeURIComponent('devices/corrector');
    toHold.add(first);
    await changeHash('#devices/corrector');
    await changeHash('#devices/bpm');
    expect(reader().dataset.conceptId).toBe('devices/bpm');

    await release(first);

    expect(reader().dataset.conceptId).toBe('devices/bpm');
  });
});
