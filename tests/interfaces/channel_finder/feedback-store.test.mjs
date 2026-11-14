// @ts-check
/**
 * The Feedback view over the feedback store (feedback.js): a store that is
 * switched off says so in operator language, naming no dotted config key; an
 * available store lists its entries and offers promotion out of the capture
 * queue; the Last Activity column reads the facility clock.
 *   npx vitest run tests/interfaces/channel_finder/feedback-store.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). The design system's toast is mocked.
 */

import { test, expect, vi, afterEach } from 'vitest';

vi.mock('/design-system/js/toast.js', () => ({
  showToast: vi.fn(),
}));

import {
  mountFeedback,
  unmountFeedback,
} from '../../../src/osprey/interfaces/channel_finder/static/js/feedback.js';
import {
  FACILITY_ZONE,
  VIEWER_ZONE,
  stampFacilityZone,
  zoneName,
} from '../_support/facility-zone.mjs';

afterEach(() => {
  unmountFeedback();
  vi.unstubAllGlobals();
  stampFacilityZone(null);
});

/**
 * Answer each GET with the payload registered for its path. An unregistered
 * path resolves not-ok, so a view that reaches for an endpoint this fixture
 * did not intend to offer fails loudly rather than silently reading `{}`.
 * @param {Record<string, any>} routes
 */
function stubApi(routes) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ path) => {
      if (path in routes) return { ok: true, json: async () => routes[path] };
      return {
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: async () => ({ detail: `no stub: ${path}` }),
      };
    })
  );
}

/** @returns {HTMLElement} a fresh mount point. */
function freshContainer() {
  document.body.innerHTML = '<div id="view"></div>';
  return /** @type {HTMLElement} */ (document.getElementById('view'));
}

const PENDING_ITEM = {
  id: 'item-1',
  query: 'all horizontal correctors',
  tool_name: 'mcp__channel-finder__build_channels',
  tool_response: JSON.stringify({ total: 2, channels: ['SR:C01:BPM1', 'SR:C02:BPM1'] }),
  channel_count: 2,
  selections: {},
  captured_at: '2026-08-27T10:00:00Z',
};

test('a store-backed paradigm with feedback off gets the switched-off state, no config key', async () => {
  const container = freshContainer();
  stubApi({
    '/api/feedback/status': {
      available: false,
      paradigm: 'hierarchical',
      entry_count: 0,
      store_path: null,
    },
  });

  mountFeedback(container);
  await vi.waitFor(() =>
    expect(container.querySelector('.osprey-state-message')?.textContent).toBe(
      'Feedback is switched off for this deployment'
    )
  );

  expect(container.innerHTML).not.toMatch(/channel_finder\./);
  expect(container.innerHTML).not.toContain('<code>');
});

test('an available store still lists entries and offers promotion', async () => {
  const container = freshContainer();
  stubApi({
    '/api/feedback/status': {
      available: true,
      paradigm: 'hierarchical',
      entry_count: 1,
      store_path: '/tmp/fb.json',
    },
    '/api/feedback': {
      entries: [
        {
          key: 'k1',
          query: 'bpms',
          facility: 'ERF',
          success_count: 3,
          failure_count: 1,
          last_activity: '2026-08-27T10:00:00Z',
        },
      ],
    },
    '/api/pending-reviews/status': { available: true, item_count: 1 },
    '/api/pending-reviews': { items: [PENDING_ITEM] },
  });

  mountFeedback(container);
  await vi.waitFor(() => expect(container.querySelector('.fb-table-row')).not.toBeNull());

  expect(container.innerHTML).toContain('bpms');
  expect(container.querySelector('.fb-pending-approve')).not.toBeNull();
  expect(container.querySelector('.fb-pending-dismiss')).not.toBeNull();
});

/** The store-backed routes of one entry, as an available store answers them. */
const STORE_ROUTES = {
  '/api/feedback/status': {
    available: true,
    paradigm: 'hierarchical',
    entry_count: 1,
    store_path: '/tmp/fb.json',
  },
  '/api/feedback': {
    entries: [
      {
        key: 'k1',
        query: 'bpms',
        facility: 'ERF',
        success_count: 3,
        failure_count: 1,
        last_activity: '2026-08-27T10:00:00Z',
      },
    ],
  },
  '/api/pending-reviews/status': { available: true, item_count: 1 },
  '/api/pending-reviews': { items: [PENDING_ITEM] },
};

test('the Last Activity column reads the facility clock and names its zone', async () => {
  stampFacilityZone(FACILITY_ZONE);
  const container = freshContainer();
  stubApi(STORE_ROUTES);

  mountFeedback(container);
  await vi.waitFor(() => expect(container.querySelector('.fb-table-row')).not.toBeNull());

  const headers = container.querySelectorAll('.data-table th');
  expect(headers[4].textContent).toBe(`Last Activity (${zoneName(FACILITY_ZONE)})`);
  const cells = /** @type {Element} */ (container.querySelector('.fb-table-row')).querySelectorAll(
    'td'
  );
  const expected = new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: FACILITY_ZONE,
  }).format(new Date('2026-08-27T10:00:00Z'));
  expect(cells[4].textContent).toBe(expected);
});

test('a viewer on the facility clock sees a plain Last Activity header', async () => {
  stampFacilityZone(VIEWER_ZONE);
  const container = freshContainer();
  stubApi(STORE_ROUTES);

  mountFeedback(container);
  await vi.waitFor(() => expect(container.querySelector('.fb-table-row')).not.toBeNull());

  expect(container.querySelectorAll('.data-table th')[4].textContent).toBe('Last Activity');
});
