// @ts-check
/**
 * Regression for the CF-1 null-guard fix (stats-badges.js): after deleting the
 * eqeqeq exemption, the `!= null` guards became strict `!== null && !== undefined`.
 * A statistics payload missing `total_channels` must NOT crash (the naive
 * `!== null`-only rewrite would let `undefined` reach `.toLocaleString()`), and
 * the missing value must be skipped rather than rendered as the string "undefined".
 *   npx vitest run tests/interfaces/channel_finder/stats-badges-guard.test.mjs
 */

import { test, expect, vi, afterEach } from 'vitest';

import { refreshStatsBadges } from '../../../src/osprey/interfaces/channel_finder/static/js/stats-badges.js';

afterEach(() => {
  vi.unstubAllGlobals();
});

/**
 * Stub global fetch so fetchJSON('/api/statistics') resolves to `payload`.
 * @param {Record<string, any>} payload
 */
function stubStatistics(payload) {
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => payload })));
}

test('a payload missing total_channels does not throw and skips the missing badge', async () => {
  document.body.innerHTML = '<div id="stats-badges"></div>';
  const container = /** @type {HTMLElement} */ (document.getElementById('stats-badges'));

  // total_channels is absent (undefined) — the value that feeds .toLocaleString().
  stubStatistics({ total_systems: 5, total_families: 10 });

  await refreshStatsBadges();  // must not throw

  expect(container.innerHTML).not.toContain('undefined');
  // The missing total_channels badge is skipped (its 'channels' label is absent)...
  expect(container.innerHTML).not.toContain('channels');
  // ...while present values still render.
  expect(container.innerHTML).toContain('systems');
  expect(container.innerHTML).toContain('5');
});

test('an explicit null value is also skipped, not rendered', async () => {
  document.body.innerHTML = '<div id="stats-badges"></div>';
  const container = /** @type {HTMLElement} */ (document.getElementById('stats-badges'));

  stubStatistics({ total_channels: null, total_systems: 3 });

  await refreshStatsBadges();

  expect(container.innerHTML).not.toContain('null');
  expect(container.innerHTML).not.toContain('channels');
  expect(container.innerHTML).toContain('systems');
});
