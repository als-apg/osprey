// @ts-check
/**
 * Regression for the feedback toolbar's button order (feedback.js _renderList).
 *
 * `.fb-toolbar` lays its right cluster out against the toolbar's right edge
 * (space-between), so a member's arrival moves everything BEFORE it and
 * nothing after it. Clear All is the conditional member — it renders only
 * while entries exist — and it used to come LAST, which shoved the persistent
 * `+ Add Entry` / `Export` pair sideways on the very click that created the
 * first entry. It must be contributed FIRST, so it comes and goes in the
 * slack to the pair's left.
 *
 *   npx vitest run tests/interfaces/channel_finder/feedback-toolbar-order.test.mjs
 */

import { test, expect, vi, afterEach } from 'vitest';

import { mountFeedback, unmountFeedback } from '../../../src/osprey/interfaces/channel_finder/static/js/feedback.js';

afterEach(() => {
  unmountFeedback();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

/**
 * Stub global fetch to serve the endpoints _renderList touches.
 * @param {any[]} entries
 */
function stubFeedbackApi(entries) {
  /** @type {Record<string, any>} */
  const routes = {
    '/api/feedback/status': { available: true },
    '/api/feedback': { entries },
    '/api/pending-reviews/status': { available: false },
  };
  vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
    const payload = routes[url];
    if (payload === undefined) throw new Error(`unstubbed fetch: ${url}`);
    return { ok: true, json: async () => payload };
  }));
}

/** @param {HTMLElement} container */
async function mountAndSettle(container) {
  mountFeedback(container);
  await vi.waitFor(() => {
    expect(container.querySelector('#fb-add-btn'), 'toolbar rendered').not.toBeNull();
  });
}

test('with entries, Clear All renders BEFORE the persistent Add/Export pair', async () => {
  document.body.innerHTML = '<div id="fb-root"></div>';
  const container = /** @type {HTMLElement} */ (document.getElementById('fb-root'));

  stubFeedbackApi([
    { key: 'k1', query: 'q', facility: 'als', success_count: 1, failure_count: 0, last_activity: '2026-01-01T00:00:00Z' },
  ]);
  await mountAndSettle(container);

  const cluster = /** @type {HTMLElement} */ (container.querySelector('.fb-toolbar-right'));
  const ids = Array.from(cluster.querySelectorAll('button')).map((b) => b.id);
  expect(ids).toEqual(['fb-clear-btn', 'fb-add-btn', 'fb-export-btn']);
});

test('with no entries, Add/Export sit exactly where they sit with entries', async () => {
  // The invariant behind the order: the persistent pair's position in the
  // cluster must not depend on whether the conditional member exists. Child
  // order is the mechanism; this asserts the consequence — the pair is the
  // cluster's tail in both states.
  document.body.innerHTML = '<div id="fb-root"></div>';
  const container = /** @type {HTMLElement} */ (document.getElementById('fb-root'));

  stubFeedbackApi([]);
  await mountAndSettle(container);

  const cluster = /** @type {HTMLElement} */ (container.querySelector('.fb-toolbar-right'));
  const ids = Array.from(cluster.querySelectorAll('button')).map((b) => b.id);
  expect(ids).toEqual(['fb-add-btn', 'fb-export-btn']);
  expect(container.querySelector('#fb-clear-btn')).toBeNull();
});
