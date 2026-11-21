// @ts-check
/**
 * The in-context table's line to All PVs (explore-in-context.js) is sized by
 * the mount that asked for the count.
 *   npx vitest run tests/interfaces/channel_finder/explore-in-context.test.mjs
 *
 * A filter that matches nothing asks the Count route for the directory's size
 * and redraws the line once it answers. A remount asks again; the count each
 * mount asked for may arrive in either order, and only the mounted page's own
 * count reaches its line.
 */

import { afterEach, beforeEach, expect, test, vi } from 'vitest';

import { state } from '../../../src/osprey/interfaces/channel_finder/static/js/state.js';
import {
  mountInContext,
  unmountInContext,
} from '../../../src/osprey/interfaces/channel_finder/static/js/explore-in-context.js';

const DIRECTORY = { label: 'All PVs', configured_services: [] };

/**
 * @param {unknown} body
 */
function jsonResponse(body) {
  return { ok: true, status: 200, json: async () => body };
}

/** The count reads, in the order they were sent. */
/** @type {((total: number) => void)[]} */
let counts = [];

beforeEach(() => {
  counts = [];
  state.directory = /** @type {any} */ (DIRECTORY);
  state.scopeQuery = 'nothing';
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url) => {
      if (url === '/api/channels') return jsonResponse({ channels: [], total: 0 });
      if (url.startsWith('/api/directory/count')) {
        return new Promise((resolve) => {
          counts.push((total) => resolve(jsonResponse({ outcome: 'found', total })));
        });
      }
      throw new Error(`unstubbed fetch: ${url}`);
    })
  );
  document.body.innerHTML = '<div id="view"></div>';
});

afterEach(() => {
  unmountInContext();
  vi.unstubAllGlobals();
  state.directory = null;
  state.scopeQuery = '';
  document.body.innerHTML = '';
});

function container() {
  return /** @type {HTMLElement} */ (document.getElementById('view'));
}

function bridgeText() {
  return document.querySelector('#ic-bridge .curated-bridge-miss a')?.textContent ?? '';
}

test("of two mounts' counts, the line shows the remounted page's when the older answers last", async () => {
  await mountInContext(container());
  await vi.waitFor(() => expect(counts).toHaveLength(1));
  unmountInContext();
  state.scopeQuery = 'nothing';
  await mountInContext(container());
  await vi.waitFor(() => expect(counts).toHaveLength(2));

  counts[1](100);
  await vi.waitFor(() => expect(bridgeText()).toContain('all 100'));

  counts[0](999);
  // The released count's path through the page is promise jobs only, so one
  // task turn drains it.
  await new Promise((r) => setTimeout(r, 0));

  expect(bridgeText()).toContain('all 100');
});
