// @ts-check
/**
 * The curated views' bridge to All PVs (curated-bridge.js and its four callers).
 *
 * With the pv_directory section configured (`state.directory` set), the two
 * curated styles that search — graph and in_context — seed their box from
 * `state.scopeQuery`, write it on every search and emit `curatedResults`, and
 * a search that matches nothing offers the All PVs search for the same words,
 * sized by the Count route. Every curated style draws a "Not here?" line to
 * All PVs under its results. Without the section none of it is drawn and the
 * Count route is never asked.
 *   npx vitest run tests/interfaces/channel_finder/curated-bridge.test.mjs
 */

import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const mocks = vi.hoisted(() => ({
  mountHierarchical: vi.fn(),
  unmountHierarchical: vi.fn(),
  mountMiddleLayer: vi.fn(),
  unmountMiddleLayer: vi.fn(),
}));

// app.js boots the whole panel on import; the two browse renderers are mocked
// so explore.js is exercised on its own.
vi.mock('../../../src/osprey/interfaces/channel_finder/static/js/app.js', () => ({
  showToast: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/channel_finder/static/js/stats-badges.js', () => ({
  refreshStatsBadges: vi.fn(),
}));
vi.mock('/design-system/js/clipboard.js', () => ({ copyText: vi.fn(async () => true) }));
vi.mock('/design-system/js/frame-params.js', () => ({ isEmbedded: () => false }));
vi.mock('../../../src/osprey/interfaces/channel_finder/static/js/explore-hierarchical.js', () => ({
  mountHierarchical: mocks.mountHierarchical,
  unmountHierarchical: mocks.unmountHierarchical,
  setShowDescriptions: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/channel_finder/static/js/explore-middle-layer.js', () => ({
  mountMiddleLayer: mocks.mountMiddleLayer,
  unmountMiddleLayer: mocks.unmountMiddleLayer,
  setShowDescriptions: vi.fn(),
}));

import { state } from '../../../src/osprey/interfaces/channel_finder/static/js/state.js';
import {
  mountExplore,
  unmountExplore,
} from '../../../src/osprey/interfaces/channel_finder/static/js/explore.js';

const COUNT_PATH = '/api/directory/count';
const SEARCH_PATH = '/api/graph/search';
const DIRECTORY = {
  label: 'ChannelFinder',
  configured_services: [],
  canopy: false,
  ariel: false,
};
const SIZE = 4321;

const ROWS = [{ fullPv: 'SR01C___QFA____AM00', device: 'SR01C___QFA____', edges: [], signals: [] }];

/** @param {number} total */
function graphSearch(total) {
  return {
    total,
    devices: total ? 1 : 0,
    page: 1,
    pages: total ? 1 : 0,
    page_size: 50,
    truncated: false,
    rows: total ? ROWS : [],
    facets: {},
    empty: false,
    suggestions: [],
  };
}

const CHANNELS = [
  { name: 'BeamCurrent', address: 'SR:DCCT', description: 'ring current' },
  { name: 'Vacuum', address: 'SR:VAC', description: 'pressure' },
];

/**
 * Stub `fetch`: the graph's ontology and search, in_context's channel list and
 * the directory count.
 * @param {(url: string) => number} graphTotal - Hits the graph search answers for a url.
 */
function stubFetch(graphTotal = () => 1) {
  const fn = vi.fn(async (/** @type {any} */ input) => {
    const url = String(input);
    /** @type {any} */
    let body = {};
    if (url.startsWith('/api/graph/ontology'))
      body = { classes: [], relationship_types: [], truncated: false, empty: false };
    else if (url.startsWith(SEARCH_PATH)) body = graphSearch(graphTotal(url));
    else if (url.startsWith('/api/channels')) body = { channels: CHANNELS, total: 2 };
    else if (url.startsWith(COUNT_PATH))
      body = { outcome: 'found', total: SIZE, source: { label: 'ChannelFinder' } };
    return { ok: true, status: 200, statusText: '', json: async () => body };
  });
  vi.stubGlobal('fetch', fn);
  return fn;
}

/** @param {any} fetchMock @param {string} path */
function callsTo(fetchMock, path) {
  return fetchMock.mock.calls
    .map((/** @type {any[]} */ call) => String(call[0]))
    .filter((/** @type {string} */ url) => url.startsWith(path));
}

const TOOLBAR_HTML = (() => {
  const html = readFileSync(
    join(import.meta.dirname, '../../../src/osprey/interfaces/channel_finder/static/index.html'),
    'utf8'
  );
  const match = html.match(/<header class="osprey-panel-toolbar"[\s\S]*?<\/header>/);
  if (!match) throw new Error('index.html carries no panel toolbar');
  return match[0];
})();

/** @returns {HTMLElement} */
function container() {
  return /** @type {HTMLElement} */ (document.getElementById('view'));
}

/** @param {string} pipeline */
async function mount(pipeline) {
  state.pipelineType = pipeline;
  mountExplore(container());
  await vi.waitFor(() => {
    if (pipeline === 'graph' && !container().querySelector('#graph-finder-table'))
      throw new Error('graph not drawn');
    if (
      pipeline === 'in_context' &&
      !container().querySelector('#ic-table-area table, .empty-state')
    )
      throw new Error('table not drawn');
  });
}

/** @returns {HTMLInputElement} */
function box(pipeline = state.pipelineType) {
  const id = pipeline === 'graph' ? 'cf-search' : 'ic-filter';
  return /** @type {HTMLInputElement} */ (document.getElementById(id));
}

/** @param {string} value */
function type(value) {
  const input = box();
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

/** @returns {HTMLAnchorElement[]} */
function bridgeLinks() {
  return [...container().querySelectorAll('.curated-bridge a')].map(
    (a) => /** @type {HTMLAnchorElement} */ (a)
  );
}

/** @type {any[]} */
let reported = [];
/** @param {any} payload */
const onCurated = (payload) => reported.push(payload);

beforeEach(() => {
  localStorage.clear();
  document.body.innerHTML = `${TOOLBAR_HTML}<div id="view"></div>`;
  state.directory = { ...DIRECTORY };
  state.scopeQuery = '';
  reported = [];
  state.on('curatedResults', onCurated);
});

afterEach(() => {
  unmountExplore();
  state.off('curatedResults', onCurated);
  state.directory = null;
  state.scopeQuery = '';
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe('graph', () => {
  test('the shared words seed the toolbar search and the first request', async () => {
    const fetchMock = stubFetch();
    state.scopeQuery = 'quad';
    await mount('graph');
    expect(box().value).toBe('quad');
    expect(callsTo(fetchMock, SEARCH_PATH)[0]).toBe(`${SEARCH_PATH}?q=quad&page=1`);
    expect(reported).toEqual([{ query: 'quad', total: 1 }]);
  });

  test('hits carry "Not here?" with the same words', async () => {
    stubFetch();
    state.scopeQuery = 'quad';
    await mount('graph');
    const line = container().querySelector('.curated-bridge');
    expect(line?.textContent).toBe('Not here? Search ChannelFinder');
    expect(bridgeLinks()[0].getAttribute('href')).toBe('#/allpvs?q=quad');
  });

  test('a miss offers the All PVs search, sized by the Count route', async () => {
    const fetchMock = stubFetch((url) => (url.includes('q=zzz') ? 0 : 1));
    await mount('graph');
    vi.useFakeTimers();
    type('zzz top');
    await vi.advanceTimersByTimeAsync(500);
    vi.useRealTimers();
    await vi.waitFor(() => {
      expect(container().querySelector('.curated-bridge-miss a')?.textContent).toBe(
        `Search all ${SIZE.toLocaleString()} PVs for "zzz top" →`
      );
    });
    expect(bridgeLinks()[0].getAttribute('href')).toBe('#/allpvs?q=zzz%20top');
    expect(callsTo(fetchMock, COUNT_PATH)).toEqual([COUNT_PATH]);
    // The words typed here are the words All PVs opens with.
    expect(state.scopeQuery).toBe('zzz top');
    expect(reported.at(-1)).toEqual({ query: 'zzz top', total: 0 });
  });
});

describe('in_context', () => {
  test('the shared words seed the filter', async () => {
    stubFetch();
    state.scopeQuery = 'vacuum';
    await mount('in_context');
    expect(box().value).toBe('vacuum');
    expect(container().querySelectorAll('#ic-table-area tbody tr')).toHaveLength(1);
    expect(reported).toEqual([{ query: 'vacuum', total: 1 }]);
    expect(container().querySelector('.curated-bridge')?.textContent).toBe(
      'Not here? Search ChannelFinder'
    );
    expect(bridgeLinks()[0].getAttribute('href')).toBe('#/allpvs?q=vacuum');
  });

  test('a miss offers the All PVs search and the words seed All PVs', async () => {
    const fetchMock = stubFetch();
    await mount('in_context');
    type('kicker');
    await vi.waitFor(() => {
      expect(container().querySelector('.curated-bridge-miss a')?.textContent).toBe(
        `Search all ${SIZE.toLocaleString()} PVs for "kicker" →`
      );
    });
    expect(bridgeLinks()[0].getAttribute('href')).toBe('#/allpvs?q=kicker');
    expect(state.scopeQuery).toBe('kicker');
    expect(reported.at(-1)).toEqual({ query: 'kicker', total: 0 });
    expect(callsTo(fetchMock, COUNT_PATH)).toHaveLength(1);
  });
});

describe.each(['hierarchical', 'middle_layer'])('%s', (pipeline) => {
  test('"Not here?" sits below the renderer with the shared words', async () => {
    stubFetch();
    state.scopeQuery = 'bpm';
    await mount(pipeline);
    const line = container().querySelector('#explore-content + .curated-bridge');
    expect(line?.textContent).toBe('Not here? Search ChannelFinder');
    expect(bridgeLinks()[0].getAttribute('href')).toBe('#/allpvs?q=bpm');
  });
});

describe('without the section', () => {
  test.each(['graph', 'in_context', 'hierarchical', 'middle_layer'])(
    '%s draws no bridge, keeps the box empty and asks no count',
    async (pipeline) => {
      state.directory = null;
      state.scopeQuery = 'stale';
      const fetchMock = stubFetch(() => 0);
      await mount(pipeline);
      if (pipeline === 'graph' || pipeline === 'in_context') {
        expect(box().value).toBe('');
        type('nothing');
        await new Promise((r) => setTimeout(r, 400));
      }
      expect(container().querySelector('.curated-bridge')).toBeNull();
      expect(callsTo(fetchMock, COUNT_PATH)).toEqual([]);
      expect(reported).toEqual([]);
      expect(state.scopeQuery).toBe('stale');
    }
  );
});
