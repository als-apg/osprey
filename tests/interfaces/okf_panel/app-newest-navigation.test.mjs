// @ts-check
/**
 * The KNOWLEDGE panel's newest navigation wins (app.js): a reader load that a
 * later navigation superseded draws nothing when its answer arrives late, so
 * the pane, the history and the page-ready mark all follow the newest one:
 *   npx vitest run tests/interfaces/okf_panel/app-newest-navigation.test.mjs
 *
 * Each answer the test needs to arrive late is held until the test releases
 * it. app.js boots on evaluation, so the DOM (with `data-graph` set and the
 * page-ready mark pending, as the server renders them), `marked`, a sanitiser
 * and `fetch` are in place before the one dynamic import below. The tests run
 * in order on that one page.
 */

import { beforeAll, describe, expect, test, vi } from 'vitest';

import { payload } from './graph-fixture.mjs';
import { mountShell } from './page-shell.mjs';

/**
 * @param {unknown} body
 */
function jsonResponse(body) {
  return { ok: true, status: 200, json: async () => body };
}

/**
 * @typedef {object} Held
 * @property {Promise<unknown>} answer
 * @property {(value: unknown) => void} release
 * @property {(error: Error) => void} fail
 */

/** Answers held back until the test releases them, by URL; each is used once. @type {Map<string, Held>} */
const held = new Map();

/**
 * Hold the next answer to *url* until the returned handle releases or fails it.
 * @param {string} url
 * @returns {Held}
 */
function hold(url) {
  /** @type {(value: unknown) => void} */
  let release = () => {};
  /** @type {(error: Error) => void} */
  let fail = () => {};
  const answer = new Promise((resolve, reject) => {
    release = resolve;
    fail = reject;
  });
  const handle = { answer, release, fail };
  held.set(url, handle);
  return handle;
}

/**
 * @param {string} url
 */
function answerFor(url) {
  if (url === '/api/concepts') {
    return jsonResponse({
      groups: [
        {
          id: 'devices',
          label: 'devices',
          concepts: [
            { id: 'devices/bpm', title: 'BPM' },
            { id: 'devices/magnet', title: 'Magnet' },
          ],
        },
      ],
    });
  }
  if (url === '/api/bundle_health') return jsonResponse({ ok: true, counts: {}, total: 0 });
  if (url === '/api/structure') return jsonResponse({ markdown: '# Structure' });
  if (url === '/api/graph') return jsonResponse(payload());
  if (url.startsWith('/api/concept?id=')) {
    const id = decodeURIComponent(url.slice('/api/concept?id='.length));
    return jsonResponse({ id, frontmatter: { title: id }, body: 'Body of ' + id });
  }
  throw new Error(`unstubbed fetch: ${url}`);
}

/**
 * @param {string} url
 */
async function fakeFetch(url) {
  const late = held.get(url);
  if (late) {
    held.delete(url);
    await late.answer;
  }
  return answerFor(url);
}

async function flush() {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
}

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

const reader = () => $('reader-content');

/**
 * Change the fragment and make sure a `hashchange` reaches the panel.
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

/** @param {string} view */
async function clickView(view) {
  /** @type {HTMLElement} */ ($('okf-views').querySelector(`[data-view="${view}"]`)).click();
  await flush();
}

class FakeResizeObserver {
  observe() {}
  disconnect() {}
}

/** The start-up overview's answer, held until the first test releases it. */
let startUpOverview = /** @type {Held} */ (/** @type {unknown} */ (null));

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
  startUpOverview = hold('/api/structure');
  await import('../../../src/osprey/interfaces/okf_panel/static/js/app.js');
  await flush();
});

describe('okf panel: the newest navigation wins', () => {
  test('a page opened while the start-up overview loads stays when the overview answers', async () => {
    await changeHash('#devices/bpm');
    expect(reader().dataset.conceptId).toBe('devices/bpm');

    startUpOverview.release(undefined);
    await flush();

    expect(reader().dataset.conceptId).toBe('devices/bpm');
    expect(reader().querySelector('.concept-title')?.textContent).toBe('devices/bpm');
    expect(history.state).toEqual({ id: 'devices/bpm' });
    expect(document.documentElement.getAttribute('data-page-ready')).toBe('true');
  });

  test('a page superseded by the overview draws nothing when it answers', async () => {
    const magnet = hold(conceptFetch('devices/magnet'));
    await changeHash('#devices/magnet');
    await changeHash('#__structure');
    expect(reader().hasAttribute('data-concept-id')).toBe(false);

    magnet.release(undefined);
    await flush();

    expect(reader().hasAttribute('data-concept-id')).toBe(false);
    expect(reader().textContent).toContain('Structure');
    expect(history.state).toEqual({ id: '__structure' });
  });

  test('a superseded load that fails draws no error', async () => {
    const magnet = hold(conceptFetch('devices/magnet'));
    await changeHash('#devices/magnet');
    await changeHash('#devices/bpm');

    magnet.fail(new TypeError('Failed to fetch'));
    await flush();

    expect(reader().dataset.conceptId).toBe('devices/bpm');
    expect(reader().textContent).not.toContain('did not load');
  });

  test('a page the Graph view interrupted loads again when Knowledge is shown', async () => {
    const magnet = hold(conceptFetch('devices/magnet'));
    await changeHash('#devices/magnet');
    await clickView('graph');

    magnet.release(undefined);
    await flush();
    expect($('graph-view').hidden).toBe(false);

    await clickView('knowledge');
    expect($('knowledge-view').hidden).toBe(false);
    expect(reader().dataset.conceptId).toBe('devices/magnet');
    expect(history.state).toEqual({ id: 'devices/magnet' });
  });
});
