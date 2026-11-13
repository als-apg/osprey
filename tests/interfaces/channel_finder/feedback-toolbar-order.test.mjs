// @ts-check
/**
 * The Feedback view's entries actions keep their places (feedback.js
 * _renderList, DESIGN.md §12.1): New entry, Export and Clear all sit in the
 * section header in that order whether or not entries exist. Clear all is
 * disabled, never removed, while there is nothing to clear, so the first entry
 * moves no button.
 *
 *   npx vitest run tests/interfaces/channel_finder/feedback-toolbar-order.test.mjs
 */

import { test, expect, vi, afterEach } from 'vitest';

import {
  mountFeedback,
  unmountFeedback,
} from '../../../src/osprey/interfaces/channel_finder/static/js/feedback.js';

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
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url) => {
      const payload = routes[url];
      if (payload === undefined) throw new Error(`unstubbed fetch: ${url}`);
      return { ok: true, json: async () => payload };
    })
  );
}

/** @param {HTMLElement} container */
async function mountAndSettle(container) {
  mountFeedback(container);
  await vi.waitFor(() => {
    expect(
      container.querySelector('[data-action="new"]'),
      'section header rendered'
    ).not.toBeNull();
  });
}

/**
 * The header's buttons as `action:disabled`.
 * @param {HTMLElement} container
 */
function actions(container) {
  const head = /** @type {HTMLElement} */ (container.querySelector('.fb-section-head'));
  return Array.from(head.querySelectorAll('button')).map(
    (b) => `${b.dataset.action}:${b.disabled ? 'disabled' : 'enabled'}`
  );
}

const ENTRY = {
  key: 'k1',
  query: 'q',
  facility: 'als',
  success_count: 1,
  failure_count: 0,
  last_activity: '2026-01-01T00:00:00Z',
};

test('New entry, Export and Clear all keep their positions when the first entry arrives', async () => {
  document.body.innerHTML = '<div id="fb-root"></div>';
  const container = /** @type {HTMLElement} */ (document.getElementById('fb-root'));

  stubFeedbackApi([]);
  await mountAndSettle(container);
  expect(actions(container)).toEqual(['new:enabled', 'export:enabled', 'clear:disabled']);

  stubFeedbackApi([ENTRY]);
  await mountAndSettle(container);
  await vi.waitFor(() => expect(container.querySelector('.fb-table-row')).not.toBeNull());
  expect(actions(container)).toEqual(['new:enabled', 'export:enabled', 'clear:enabled']);
});
