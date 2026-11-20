// @ts-check
/**
 * The KNOWLEDGE panel started on the Graph view's fragment stays on the Graph
 * view (app.js), and its Knowledge view loads the overview when it is first
 * shown:
 *   npx vitest run tests/interfaces/okf_panel/app-graph-start-up.test.mjs
 *
 * app.js boots on evaluation, so the DOM (with `data-graph` set and the
 * page-ready mark pending, as the server renders them), the fragment, `marked`,
 * a sanitiser and `fetch` are in place before the one dynamic import below.
 * The tests run in order on that one page.
 */

import { beforeAll, describe, expect, test, vi } from 'vitest';

import { payload } from './graph-fixture.mjs';
import { mountShell } from './page-shell.mjs';

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
      ],
    });
  }
  if (url === '/api/bundle_health') return jsonResponse({ ok: true, counts: {}, total: 0 });
  if (url === '/api/structure') return jsonResponse({ markdown: '# Structure' });
  if (url === '/api/graph') return jsonResponse(payload());
  throw new Error(`unstubbed fetch: ${url}`);
}

async function flush() {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
}

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

/** @param {string} view */
async function clickView(view) {
  /** @type {HTMLElement} */ ($('okf-views').querySelector(`[data-view="${view}"]`)).click();
  await flush();
}

class FakeResizeObserver {
  observe() {}
  disconnect() {}
}

beforeAll(async () => {
  mountShell();
  $('app').dataset.graph = 'true';
  document.documentElement.setAttribute('data-page-ready', 'pending');
  localStorage.clear();
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
  vi.stubGlobal('marked', { parse: (/** @type {string} */ md) => md });
  vi.stubGlobal('DOMPurify', {
    sanitize: (/** @type {string} */ html) => html,
    addHook: () => {},
    removeHook: () => {},
  });
  vi.stubGlobal('fetch', vi.fn(fakeFetch));
  history.replaceState(null, '', location.pathname);
  location.hash = '#__graph';
  await import('../../../src/osprey/interfaces/okf_panel/static/js/app.js');
  await flush();
});

describe('okf panel started on the Graph view', () => {
  test('the Graph view shows, and the overview is not loaded', () => {
    expect($('graph-view').hidden).toBe(false);
    expect($('knowledge-view').hidden).toBe(true);
    expect($('graph-stage').querySelector('.okf-graph-node')).not.toBeNull();
    expect(location.hash).toBe('#__graph');
    expect(history.state).toEqual({ id: '__graph', node: null });
    expect(fetched).not.toContain('/api/structure');
    expect(document.documentElement.getAttribute('data-page-ready')).toBe('true');
  });

  test('the Knowledge view loads the overview when it is first shown', async () => {
    await clickView('knowledge');

    expect($('knowledge-view').hidden).toBe(false);
    expect($('graph-view').hidden).toBe(true);
    expect(fetched.filter((url) => url === '/api/structure')).toHaveLength(1);
    expect($('reader-content').textContent).toContain('Structure');
    expect(history.state).toEqual({ id: '__structure' });
  });
});
