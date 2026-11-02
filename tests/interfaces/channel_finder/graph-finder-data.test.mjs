// @ts-check
/**
 * Unit tests for the graph finder's reads (graph-finder-data.js): one request
 * in flight per kind, and a reply dropped once the view has moved on.
 *   npx vitest run tests/interfaces/channel_finder/graph-finder-data.test.mjs
 */

import { afterEach, expect, test, vi } from 'vitest';

import { createFlights } from '../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-data.js';

/**
 * Replace `fetch` with one whose replies the test releases by hand.
 * @returns {{release: (index: number, body: object) => void, signals: AbortSignal[]}}
 */
function heldFetch() {
  /** @type {Array<(r: Response) => void>} */
  const pending = [];
  /** @type {AbortSignal[]} */
  const signals = [];
  vi.stubGlobal(
    'fetch',
    vi.fn((_url, init) => {
      signals.push(init.signal);
      return new Promise((resolve) => pending.push(resolve));
    })
  );
  return {
    release: (index, body) => pending[index](new Response(JSON.stringify(body), { status: 200 })),
    signals,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

test('a request answers with its payload while its pane stays mounted', async () => {
  const fetches = heldFetch();
  const pane = {};
  const flights = createFlights(() => pane);

  const reply = flights.request('search', '/api/graph/search?q=a');
  fetches.release(0, { results: [1] });

  expect(await reply).toEqual({ ok: true, data: { results: [1] } });
});

test('a newer request of the same kind aborts the older one and drops its reply', async () => {
  const fetches = heldFetch();
  const pane = {};
  const flights = createFlights(() => pane);

  const first = flights.request('search', '/api/graph/search?q=a');
  const second = flights.request('search', '/api/graph/search?q=ab');
  fetches.release(0, { results: ['stale'] });
  fetches.release(1, { results: ['fresh'] });

  expect(fetches.signals[0].aborted).toBe(true);
  expect(await first).toBeNull();
  expect(await second).toEqual({ ok: true, data: { results: ['fresh'] } });
});

test('kinds fly independently: a search does not abort the device read', async () => {
  const fetches = heldFetch();
  const pane = {};
  const flights = createFlights(() => pane);

  const device = flights.request('device', '/api/graph/device?uri=x');
  const search = flights.request('search', '/api/graph/search?q=a');
  fetches.release(0, { uri: 'x' });
  fetches.release(1, { results: [] });

  expect(fetches.signals[0].aborted).toBe(false);
  expect(await device).toEqual({ ok: true, data: { uri: 'x' } });
  expect(await search).toEqual({ ok: true, data: { results: [] } });
});

test('abort retires the kind: a reply already on its way is dropped', async () => {
  const fetches = heldFetch();
  const pane = {};
  const flights = createFlights(() => pane);

  const reply = flights.request('device', '/api/graph/device?uri=x');
  flights.abort('device');
  fetches.release(0, { uri: 'x' });

  expect(fetches.signals[0].aborted).toBe(true);
  expect(await reply).toBeNull();
});

test('a reply that lands after its pane was unmounted or replaced is dropped', async () => {
  const fetches = heldFetch();
  /** @type {object|null} */
  let pane = {};
  const flights = createFlights(() => pane);

  const unmounted = flights.request('ontology', '/api/graph/ontology');
  pane = null;
  fetches.release(0, { classes: [] });
  expect(await unmounted).toBeNull();

  pane = {};
  const replaced = flights.request('ontology', '/api/graph/ontology');
  pane = {};
  fetches.release(1, { classes: [] });
  expect(await replaced).toBeNull();
});
