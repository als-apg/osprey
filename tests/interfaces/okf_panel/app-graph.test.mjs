// @ts-check
/**
 * The KNOWLEDGE panel moves between its Knowledge view and its Graph view
 * (app.js), and its URL fragment says which one is open:
 *   npx vitest run tests/interfaces/okf_panel/app-graph.test.mjs
 *
 * app.js boots on evaluation, so the DOM (with `data-graph` set, as the
 * server renders it for a panel that serves a graph), `marked`, a sanitiser
 * and `fetch` are in place before the one dynamic import below. The tests run
 * in order on that one page.
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
  if (url.startsWith('/api/concept?id=')) {
    const id = decodeURIComponent(url.slice('/api/concept?id='.length));
    return jsonResponse({
      id,
      frontmatter: { title: id },
      body: '<a href="facility:M:Q1:SP">setpoint</a> <a href="facility:NOPE">gone</a>',
      facility_references: [
        {
          start: 0,
          end: 1,
          target: 'M:Q1:SP',
          kinds: ['channel'],
          label: null,
          records: [{ kind: 'channel', id: 'M:Q1:SP' }],
        },
        { start: 2, end: 3, target: 'NOPE', kinds: [], label: null, records: null },
      ],
    });
  }
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

/** @param {string} key */
function clickNode(key, type = 'click') {
  const node = /** @type {Element} */ (
    $('graph-stage').querySelector(`.okf-graph-node[data-key="${key}"]`)
  );
  node.dispatchEvent(new MouseEvent(type, { bubbles: true }));
}

class FakeResizeObserver {
  observe() {}
  disconnect() {}
}

beforeAll(async () => {
  mountShell();
  $('app').dataset.graph = 'true';
  localStorage.clear();
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
  vi.stubGlobal('marked', { parse: (/** @type {string} */ md) => md });
  vi.stubGlobal('DOMPurify', { sanitize: (/** @type {string} */ html) => html });
  vi.stubGlobal('fetch', vi.fn(fakeFetch));
  location.hash = '#devices/bpm';
  await import('../../../src/osprey/interfaces/okf_panel/static/js/app.js');
  await flush();
});

describe('okf panel views', () => {
  test('the panel opens on the Knowledge view and does not fetch the graph', () => {
    expect($('okf-views').hidden).toBe(false);
    expect($('graph-view').hidden).toBe(true);
    expect(fetched).not.toContain('/api/graph');
  });

  test('the switcher opens the graph on its overview, under its own fragment', async () => {
    await clickView('graph');
    expect($('graph-view').hidden).toBe(false);
    expect($('knowledge-view').hidden).toBe(true);
    expect(fetched).toContain('/api/graph');
    expect(location.hash).toBe('#__graph');
    expect(history.state).toEqual({ id: '__graph', node: null });
    expect($('graph-context').textContent).toBe('Overview');
  });

  test('the fragment follows the focus without adding history entries', async () => {
    const entries = history.length;
    clickNode('g:channel');
    /** @type {HTMLElement} */ ($('graph-inspector').querySelector('.okf-graph-row')).click();
    await flush();
    expect(location.hash).toBe('#__graph/' + encodeURIComponent('c:M:BPM1:X'));
    expect(history.state).toEqual({ id: '__graph', node: 'c:M:BPM1:X' });
    expect(history.length).toBe(entries);
  });

  test('the search field searches the open view', async () => {
    const input = /** @type {HTMLInputElement} */ ($('search-input'));
    input.value = 'q1';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    await new Promise((r) => setTimeout(r, 400));
    await flush();
    expect($('graph-hits').hidden).toBe(false);
    expect($('graph-hits').textContent).toContain('M/Q1');
    expect(fetched.some((url) => url.startsWith('/api/search'))).toBe(false);
  });

  test('going back to Knowledge restores the page, and empties the search', async () => {
    await clickView('knowledge');
    expect($('knowledge-view').hidden).toBe(false);
    expect(location.hash).toBe('#devices/bpm');
    expect(/** @type {HTMLInputElement} */ ($('search-input')).value).toBe('');
    expect($('graph-hits').hidden).toBe(true);
  });

  test('the graph kept its place', async () => {
    await clickView('graph');
    expect(location.hash).toBe('#__graph/' + encodeURIComponent('c:M:BPM1:X'));
    expect($('graph-context').textContent).toBe('Neighbourhood of M:BPM1:X');
  });

  test('Knowledge page opens the page in the Knowledge view', async () => {
    clickNode('2');
    const open = [...$('graph-inspector').querySelectorAll('button')].find(
      (button) => button.textContent === 'Knowledge page'
    );
    /** @type {HTMLElement} */ (open).click();
    await flush();
    expect($('knowledge-view').hidden).toBe(false);
    expect(history.state).toEqual({ id: 'devices/bpm' });
  });

  test('a facility link opens the graph on its record; a dangling one does not', async () => {
    const links = $('reader-content').querySelectorAll('a');
    expect(links[0].classList.contains('okf-facility-link--graph')).toBe(true);
    expect(links[0].getAttribute('title')).toContain('Click: show in the graph');
    expect(links[1].classList.contains('okf-facility-link--graph')).toBe(false);

    links[1].click();
    await flush();
    expect($('knowledge-view').hidden).toBe(false);

    links[0].click();
    await flush();
    expect($('graph-view').hidden).toBe(false);
    expect($('graph-context').textContent).toBe('Neighbourhood of M:Q1:SP');
    expect(location.hash).toBe('#__graph/' + encodeURIComponent('c:M:Q1:SP'));
  });

  test('a graph fragment set from outside opens that neighbourhood', async () => {
    await clickView('knowledge');
    location.hash = '#__graph/' + encodeURIComponent('d:M/Q1');
    await flush();
    window.dispatchEvent(new HashChangeEvent('hashchange'));
    await flush();
    expect($('graph-view').hidden).toBe(false);
    expect($('graph-context').textContent).toBe('Neighbourhood of M/Q1');
  });

  test('history navigation reopens each view', async () => {
    window.dispatchEvent(new PopStateEvent('popstate', { state: { id: 'devices/bpm' } }));
    await flush();
    expect($('knowledge-view').hidden).toBe(false);
    window.dispatchEvent(new PopStateEvent('popstate', { state: { id: '__graph', node: null } }));
    await flush();
    expect($('graph-view').hidden).toBe(false);
    expect($('graph-context').textContent).toBe('Overview');
  });

  test('going back to the entry the panel opened on leaves the Graph view', async () => {
    window.dispatchEvent(new PopStateEvent('popstate', { state: { id: '__graph', node: null } }));
    await flush();
    expect($('graph-view').hidden).toBe(false);
    window.dispatchEvent(new PopStateEvent('popstate', { state: null }));
    await flush();
    expect($('knowledge-view').hidden).toBe(false);
    expect($('graph-view').hidden).toBe(true);
  });
});
