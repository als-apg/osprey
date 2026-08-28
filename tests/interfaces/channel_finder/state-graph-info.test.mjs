// @ts-check
/**
 * Boot contract for the graph provenance fields on the shared state (state.js).
 *
 * GET /api/info reports `tools` and `graph_store` on the graph paradigm only.
 * The boot in app.js must land both on the shared state, and a file-backed
 * payload — which carries neither key — must leave the empty defaults in place
 * so no consumer mistakes a missing answer for a graph-backed one.
 *   npx vitest run tests/interfaces/channel_finder/state-graph-info.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). app.js boots on import, so each boot
 * case re-imports it into a fresh module registry (vi.resetModules) with the
 * fetch stub for that case already installed.
 */

import { test, expect, vi, afterEach } from 'vitest';

const STATE_MODULE = '../../../src/osprey/interfaces/channel_finder/static/js/state.js';
const APP_MODULE = '../../../src/osprey/interfaces/channel_finder/static/js/app.js';

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

/**
 * Answer GET /api/info with `payload`; every other path (the boot also pokes
 * /api/statistics) gets an empty body.
 * @param {Record<string, any>} payload
 */
function stubInfo(payload) {
  vi.stubGlobal('fetch', vi.fn(async (/** @type {any} */ url) => ({
    ok: true,
    json: async () => (String(url).includes('/api/info') ? payload : {}),
  })));
}

/**
 * Run the app.js boot against `payload` and return the state singleton it
 * populated. The document is left empty, so the boot's view routing and stats
 * badges find no hosts and no-op — only the /api/info leg is under test.
 * @param {Record<string, any>} payload
 * @returns {Promise<any>}
 */
async function boot(payload) {
  stubInfo(payload);
  vi.resetModules();
  const { state } = await import(STATE_MODULE);
  await import(APP_MODULE);
  await vi.waitFor(() => expect(state.pipelineType).toBe(payload.pipeline_type));
  return state;
}

test('a graph /api/info payload lands its tools and graph store on the state', async () => {
  const state = await boot({
    pipeline_type: 'graph',
    metadata: { facility_name: 'ALS' },
    available_pipelines: ['graph'],
    db_path: null,
    tools: ['capabilities', 'example_queries', 'get_schema', 'read_cypher'],
    graph_store: { uri: 'bolt://localhost:7687', ttl_filename: 'als.ttl' },
  });

  expect(state.tools).toEqual([
    'capabilities', 'example_queries', 'get_schema', 'read_cypher',
  ]);
  expect(state.graphStore).toEqual({
    uri: 'bolt://localhost:7687',
    ttl_filename: 'als.ttl',
  });
});

test('a file-backed payload without the graph keys keeps the empty defaults', async () => {
  const state = await boot({
    pipeline_type: 'hierarchical',
    metadata: {},
    available_pipelines: ['hierarchical'],
    db_path: '/tmp/channels.db',
  });

  expect(state.tools).toEqual([]);
  expect(state.graphStore).toBeNull();
});

test('setGraphInfo records both values and falls back to the defaults', async () => {
  vi.resetModules();
  const { state } = await import(STATE_MODULE);

  state.setGraphInfo(['read_cypher'], { uri: 'bolt://graphdb:7687', ttl_filename: null });
  expect(state.tools).toEqual(['read_cypher']);
  expect(state.graphStore).toEqual({ uri: 'bolt://graphdb:7687', ttl_filename: null });

  // A later answer with neither field clears the previous one rather than
  // leaving a stale graph store behind.
  state.setGraphInfo(undefined, undefined);
  expect(state.tools).toEqual([]);
  expect(state.graphStore).toBeNull();
});
