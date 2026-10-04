// @ts-check
/**
 * The KNOWLEDGE panel follows a change of its own URL fragment to the concept
 * it names (app.js's `hashchange` listener):
 *   npx vitest run tests/interfaces/okf_panel/app-hash-nav.test.mjs
 *
 * The hub opens this panel on a concept by navigating its iframe to
 * `panel/okf#<concept id>`; when the panel is already loaded that is a fragment
 * navigation, so the panel must follow it. A fragment that names no concept is
 * an in-page anchor and is left to the browser.
 *
 * app.js boots on evaluation, so the DOM, `marked` and `fetch` are in place
 * before the one dynamic import below.
 */

import { beforeAll, describe, expect, test, vi } from 'vitest';

/** @type {string[]} */
const fetched = [];

/**
 * @param {unknown} body
 */
function jsonResponse(body) {
  return { ok: true, status: 200, json: async () => body };
}

/**
 * @param {string} url
 */
async function fakeFetch(url) {
  fetched.push(url);
  if (url === '/api/concepts') {
    return jsonResponse({
      groups: [
        { id: 'devices', label: 'devices', concepts: [{ id: 'devices/bpm', title: 'BPM' }] },
        {
          id: 'procedures',
          label: 'procedures',
          concepts: [{ id: 'procedures/orbit-correction', title: 'Orbit correction' }],
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

async function flush() {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * Change the fragment and make sure a `hashchange` reaches the panel. The
 * listener ignores a target `history.state` already holds, so a second event
 * for the same change is harmless.
 * @param {string} hash
 */
async function changeHash(hash) {
  location.hash = hash;
  await flush();
  window.dispatchEvent(new HashChangeEvent('hashchange'));
  await flush();
}

/**
 * @param {string} id
 */
function conceptFetch(id) {
  return '/api/concept?id=' + encodeURIComponent(id);
}

beforeAll(async () => {
  document.body.innerHTML = `
    <div id="app">
      <aside id="sidebar">
        <form id="search-form"><input id="search-input" type="search" /></form>
        <a id="structure-link" href="#__structure">Knowledge Base</a>
        <nav id="tree"></nav>
        <div id="search-results" hidden></div>
        <footer id="bundle-health" hidden></footer>
      </aside>
      <main id="reader">
        <button type="button" id="browse-all">Browse all pages</button>
        <div id="reader-content"></div>
      </main>
    </div>`;
  vi.stubGlobal('marked', { parse: (/** @type {string} */ md) => md });
  vi.stubGlobal('fetch', vi.fn(fakeFetch));
  location.hash = '#devices/bpm';
  await import('../../../src/osprey/interfaces/okf_panel/static/js/app.js');
  await flush();
});

describe('okf panel fragment navigation', () => {
  test('the boot fragment loads its concept', () => {
    expect(fetched).toContain(conceptFetch('devices/bpm'));
  });

  test('a fragment change loads the concept it names', async () => {
    await changeHash('#procedures/orbit-correction');
    expect(fetched).toContain(conceptFetch('procedures/orbit-correction'));
    expect(history.state).toEqual({ id: 'procedures/orbit-correction' });
  });

  test('an in-page anchor is left alone', async () => {
    await changeHash('#some-heading');
    expect(fetched).not.toContain(conceptFetch('some-heading'));
  });
});
