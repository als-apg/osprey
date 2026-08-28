// @ts-check
/**
 * Paradigm dispatch for the Feedback view (feedback.js).
 *
 * `/api/feedback/status` answering `available: false` has two unrelated
 * causes, and the view must not answer both the same way. Under the graph
 * paradigm there is no hint store by design, yet the capture hook still fills
 * the pending queue — so the view shows that queue instead of a lock screen,
 * and offers Dismiss without Approve (the server refuses promotion where there
 * is no feedback store). Under any other paradigm the store is simply switched
 * off, and the pane says so in operator language, naming no dotted config key.
 *   npx vitest run tests/interfaces/channel_finder/feedback-graph-mode.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). app.js is mocked so importing the
 * view does not boot the whole panel; feedback.js wants only `showToast`.
 */

import { test, expect, vi, afterEach } from 'vitest';

vi.mock('../../../src/osprey/interfaces/channel_finder/static/js/app.js', () => ({
  showToast: vi.fn(),
}));

import { mountFeedback, unmountFeedback } from '../../../src/osprey/interfaces/channel_finder/static/js/feedback.js';

afterEach(() => {
  unmountFeedback();
  vi.unstubAllGlobals();
});

/**
 * Answer each GET with the payload registered for its path. An unregistered
 * path resolves not-ok, so a view that reaches for an endpoint this fixture
 * did not intend to offer fails loudly rather than silently reading `{}`.
 * @param {Record<string, any>} routes
 */
function stubApi(routes) {
  vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ path) => {
    if (path in routes) return { ok: true, json: async () => routes[path] };
    return { ok: false, status: 404, statusText: 'Not Found', json: async () => ({ detail: `no stub: ${path}` }) };
  }));
}

/** @returns {HTMLElement} a fresh mount point. */
function freshContainer() {
  document.body.innerHTML = '<div id="view"></div>';
  return /** @type {HTMLElement} */ (document.getElementById('view'));
}

const PENDING_ITEM = {
  id: 'item-1',
  query: 'all horizontal correctors',
  tool_name: 'mcp__channel-finder__read_cypher',
  tool_response: JSON.stringify({ row_count: 2, channels: ['SR:C01:BPM1', 'SR:C02:BPM1'] }),
  channel_count: 2,
  selections: {},
  captured_at: '2026-08-27T10:00:00Z',
};

test('graph mode shows the capture queue, not the lock screen', async () => {
  const container = freshContainer();
  stubApi({
    '/api/feedback/status': { available: false, paradigm: 'graph', entry_count: 0, store_path: null },
    '/api/pending-reviews/status': { available: true, item_count: 1, can_promote: false },
    '/api/pending-reviews': { items: [PENDING_ITEM] },
  });

  mountFeedback(container);
  await vi.waitFor(() => expect(container.querySelector('.fb-pending-card')).not.toBeNull());

  // The captured search is on screen...
  expect(container.innerHTML).toContain('all horizontal correctors');
  expect(container.innerHTML).toContain('SR:C01:BPM1');
  // ...with Dismiss offered and Approve withheld: promotion would 404.
  expect(container.querySelector('.fb-pending-dismiss')).not.toBeNull();
  expect(container.querySelector('.fb-pending-approve')).toBeNull();
  // The lock screen and its hierarchical-only config key are both gone.
  expect(container.querySelector('.fb-disabled')).toBeNull();
  expect(container.innerHTML).not.toMatch(/channel_finder\./);
});

test('graph mode with an empty queue still explains itself', async () => {
  const container = freshContainer();
  stubApi({
    '/api/feedback/status': { available: false, paradigm: 'graph', entry_count: 0, store_path: null },
    '/api/pending-reviews/status': { available: true, item_count: 0, can_promote: false },
    '/api/pending-reviews': { items: [] },
  });

  mountFeedback(container);
  await vi.waitFor(() => expect(container.querySelector('.fb-paradigm-note')).not.toBeNull());

  expect(container.innerHTML).toContain('No pending reviews');
  expect(container.querySelector('.fb-disabled')).toBeNull();
  expect(container.innerHTML).not.toMatch(/channel_finder\./);
});

test('a store-backed paradigm with feedback off gets the operator pane, no config key', async () => {
  const container = freshContainer();
  stubApi({
    '/api/feedback/status': { available: false, paradigm: 'hierarchical', entry_count: 0, store_path: null },
  });

  mountFeedback(container);
  await vi.waitFor(() => expect(container.querySelector('.fb-disabled')).not.toBeNull());

  expect(container.innerHTML).not.toMatch(/channel_finder\./);
  expect(container.innerHTML).not.toContain('<code>');
  expect(container.querySelector('.fb-paradigm-note')).toBeNull();
});

test('an available store still lists entries and offers promotion', async () => {
  const container = freshContainer();
  stubApi({
    '/api/feedback/status': { available: true, paradigm: 'hierarchical', entry_count: 1, store_path: '/tmp/fb.json' },
    '/api/feedback': {
      entries: [{
        key: 'k1', query: 'bpms', facility: 'ALS',
        success_count: 3, failure_count: 1, last_activity: '2026-08-27T10:00:00Z',
      }],
    },
    '/api/pending-reviews/status': { available: true, item_count: 1, can_promote: true },
    '/api/pending-reviews': { items: [PENDING_ITEM] },
  });

  mountFeedback(container);
  await vi.waitFor(() => expect(container.querySelector('.fb-table-row')).not.toBeNull());

  expect(container.innerHTML).toContain('bpms');
  expect(container.querySelector('.fb-pending-approve')).not.toBeNull();
  expect(container.querySelector('.fb-pending-dismiss')).not.toBeNull();
});
