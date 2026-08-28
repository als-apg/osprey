// @ts-check
/**
 * Dispatch contract for the Explore view (explore.js).
 *
 * Every known pipeline type — graph included — must reach its own renderer, and
 * a pipeline type this view does not know about must land on the explicit
 * "unknown" pane, never on the in-context renderer as a silent catch-all.
 *   npx vitest run tests/interfaces/channel_finder/explore-graph.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). The four renderer modules are
 * mocked so the dispatch is tested on its own, without their network calls.
 */

import { test, expect, vi, beforeEach } from 'vitest';

vi.hoisted(() => {
  vi.stubGlobal('fetch', () => new Promise(() => {}));
});

const mocks = vi.hoisted(() => ({
  mountHierarchical: vi.fn(),
  unmountHierarchical: vi.fn(),
  mountMiddleLayer: vi.fn(),
  unmountMiddleLayer: vi.fn(),
  mountInContext: vi.fn(),
  unmountInContext: vi.fn(),
  mountGraph: vi.fn(),
  unmountGraph: vi.fn(),
}));

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

vi.mock('../../../src/osprey/interfaces/channel_finder/static/js/explore-in-context.js', () => ({
  mountInContext: mocks.mountInContext,
  unmountInContext: mocks.unmountInContext,
}));

vi.mock('../../../src/osprey/interfaces/channel_finder/static/js/explore-graph.js', () => ({
  mountGraph: mocks.mountGraph,
  unmountGraph: mocks.unmountGraph,
}));

import { state } from '../../../src/osprey/interfaces/channel_finder/static/js/state.js';
import { mountExplore, unmountExplore } from '../../../src/osprey/interfaces/channel_finder/static/js/explore.js';
import { renderSchema } from '../../../src/osprey/interfaces/channel_finder/static/js/utils.js';
import { refreshStatsBadges } from '../../../src/osprey/interfaces/channel_finder/static/js/stats-badges.js';
// app.js boots the panel on import (readyState is already 'complete' here), so
// its /api/info probe is parked on a promise that never settles: no network
// call, no console noise, and nothing of its boot lands on the shared state.
import { PIPELINE_LABELS } from '../../../src/osprey/interfaces/channel_finder/static/js/app.js';

/**
 * Every pipeline type the Explore view renders, paired with the mount/unmount
 * functions it must reach. A new paradigm adds one row here.
 * @type {{ type: string, mount: import('vitest').Mock, unmount: import('vitest').Mock }[]}
 */
const KNOWN_TYPES = [
  { type: 'hierarchical', mount: mocks.mountHierarchical, unmount: mocks.unmountHierarchical },
  { type: 'middle_layer', mount: mocks.mountMiddleLayer, unmount: mocks.unmountMiddleLayer },
  { type: 'in_context', mount: mocks.mountInContext, unmount: mocks.unmountInContext },
  { type: 'graph', mount: mocks.mountGraph, unmount: mocks.unmountGraph },
];

const ALL_MOUNTS = KNOWN_TYPES.map(t => t.mount);

/**
 * Put a pipeline type on the shared state and mount Explore into a fresh
 * document, returning the container.
 * @param {string|null} pipelineType
 * @param {string|null} [dbPath] - Database path the header may badge.
 * @returns {HTMLElement}
 */
function mountWith(pipelineType, dbPath = null) {
  state.pipelineType = pipelineType;
  state.pipelineMetadata = {};
  state.dbPath = dbPath;
  document.body.innerHTML = '<div id="explore-root"></div>';
  const container = /** @type {HTMLElement} */ (document.getElementById('explore-root'));
  mountExplore(container);
  return container;
}

/**
 * The loud pane for a pipeline type this view has no renderer for.
 * @returns {HTMLElement|null}
 */
const unknownPane = () => document.querySelector('.explore-unknown:not([data-pipeline])');

beforeEach(() => {
  vi.clearAllMocks();
  document.body.innerHTML = '';
  state.pipelineType = null;
});

test.each(KNOWN_TYPES)('pipeline type $type dispatches to its own renderer', ({ type, mount }) => {
  mountWith(type);

  expect(mount).toHaveBeenCalledTimes(1);
  // It receives the content host, not the outer container.
  expect(mount.mock.calls[0][0]).toBe(document.getElementById('explore-content'));
  // No other renderer ran, and no unknown pane was shown.
  for (const other of ALL_MOUNTS) {
    if (other !== mount) expect(other).not.toHaveBeenCalled();
  }
  expect(unknownPane()).toBeNull();
});

test.each(KNOWN_TYPES)('unmounting after $type calls that renderer\'s unmount only', ({ type, unmount }) => {
  mountWith(type);
  unmountExplore();

  expect(unmount).toHaveBeenCalledTimes(1);
  for (const row of KNOWN_TYPES) {
    if (row.unmount !== unmount) expect(row.unmount).not.toHaveBeenCalled();
  }
});

test('an unknown pipeline type mounts the unknown pane and no renderer', () => {
  mountWith('no_such_pipeline');

  const pane = unknownPane();
  expect(pane, 'unknown pane is mounted').not.toBeNull();
  expect(pane?.textContent).toContain("Unknown pipeline 'no_such_pipeline'");
  expect(pane?.textContent).toContain('the server rejected this configuration');

  // The in-context renderer is not a catch-all.
  expect(mocks.mountInContext).not.toHaveBeenCalled();
  for (const mount of ALL_MOUNTS) expect(mount).not.toHaveBeenCalled();
});

test('a null pipeline type is unknown too', () => {
  mountWith(null);

  expect(unknownPane(), 'unknown pane is mounted for a missing type').not.toBeNull();
  for (const mount of ALL_MOUNTS) expect(mount).not.toHaveBeenCalled();
});

test('the unknown pane escapes the pipeline type it echoes back', () => {
  mountWith('<img src=x onerror=alert(1)>');

  const content = /** @type {HTMLElement} */ (document.getElementById('explore-content'));
  expect(content.querySelector('img'), 'no live <img> node').toBeNull();
  const hasOnAttr = [...content.querySelectorAll('*')].some(el =>
    [...el.attributes].some(attr => attr.name.startsWith('on'))
  );
  expect(hasOnAttr, 'no on* event-handler attribute').toBe(false);
  // The value still reaches the reader, as inert text.
  expect(content.textContent).toContain('<img src=x onerror=alert(1)>');
});

test('unmounting an unknown pipeline clears the pane and calls no renderer unmount', () => {
  mountWith('no_such_pipeline');
  expect(unknownPane()).not.toBeNull();

  unmountExplore();

  expect(unknownPane(), 'pane is cleared on unmount').toBeNull();
  for (const row of KNOWN_TYPES) expect(row.unmount).not.toHaveBeenCalled();
});

// ---- Graph paradigm: a renderer like the others, with its own chrome ----

test('the graph paradigm reaches its renderer and not the unknown pane', () => {
  mountWith('graph');

  expect(mocks.mountGraph).toHaveBeenCalledTimes(1);
  // Graph is a configuration the server accepts, so nothing here may read as
  // "the server rejected this".
  expect(unknownPane(), 'not the unknown-pipeline pane').toBeNull();
  expect(document.body.textContent).not.toContain('Unknown pipeline');
});

test('the graph header keeps the paradigm subtitle', () => {
  const container = mountWith('graph');

  expect(container.querySelector('.section-subtitle')?.textContent).toBe(
    'Channels are resolved from the facility graph'
  );
});

test('the graph header shows no db-source badge \u2014 the renderer names the store', () => {
  // A db path left on the shared state by a previously mounted mode must not
  // surface here: graph channels come from the store the panel badges itself,
  // so a file path in the header would name a source this mode never reads.
  const container = mountWith('graph', '/var/lib/osprey/channels.db');
  expect(container.querySelector('.db-source-badge'), 'no db-source badge').toBeNull();

  // The badge is not gone for everyone: the file-backed modes still carry it.
  const withDb = mountWith('hierarchical', '/var/lib/osprey/channels.db');
  expect(withDb.querySelector('.db-source-badge')).not.toBeNull();
});

test('the unknown-pipeline pane keeps the error posture', () => {
  mountWith('no_such_pipeline');

  const pane = unknownPane();
  expect(pane?.classList.contains('explore-unknown--info'), 'error pane is not tinted as info').toBe(
    false
  );
  expect(pane?.getAttribute('role')).toBe('alert');
});

test('the graph pane comes with no schema diagram and no description toggle', () => {
  const container = mountWith('graph');

  expect(document.getElementById('explore-schema'), 'no schema host').toBeNull();
  expect(container.querySelector('#show-desc-toggle'), 'no description toggle').toBeNull();
});

test("renderSchema draws nothing for graph, and clears a previous mode's diagram", () => {
  document.body.innerHTML = '<div id="schema"></div>';
  const host = /** @type {HTMLElement} */ (document.getElementById('schema'));
  // Whatever a previously mounted mode left behind...
  renderSchema(host, 'hierarchical', { hierarchy_levels: ['System', 'Family'] });
  expect(host.querySelectorAll('.schema-node').length).toBeGreaterThan(0);

  // ...graph draws nothing, rather than falling through to the in-context
  // "Database -> Channels -> Name + Address" diagram.
  renderSchema(host, 'graph', { hierarchy_levels: ['System', 'Family'] });
  expect(host.innerHTML).toBe('');
  expect(host.querySelectorAll('.schema-node').length).toBe(0);
});

/**
 * The badge labels currently rendered into #stats-badges, in document order.
 * @returns {string[]}
 */
const badgeLabels = () =>
  [...document.querySelectorAll('#stats-badges .badge-label')].map(el => el.textContent ?? '');

test('the stats badges render the graph statistics in table order', async () => {
  document.body.innerHTML = '<div id="stats-badges"></div>';
  const container = /** @type {HTMLElement} */ (document.getElementById('stats-badges'));
  const fetchSpy = vi.fn(async () => ({
    ok: true,
    json: async () => ({
      total_devices: 512,
      total_channels: 2908,
      total_classes: 19,
      total_signals: 113,
      total_sections: 3,
    }),
  }));
  vi.stubGlobal('fetch', fetchSpy);
  state.pipelineType = 'graph';

  try {
    await refreshStatsBadges();
    // Graph mode counts what it can and asks for /api/statistics like every
    // other paradigm — it is no longer the one mode that skips the request.
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining('/api/statistics'),
      expect.anything()
    );
    // Exactly the five reported statistics, in the render order the module
    // owns — not the payload's key order.
    expect(container.querySelectorAll('.stats-badge')).toHaveLength(5);
    expect(badgeLabels()).toEqual(['devices', 'channels', 'classes', 'signals', 'sections']);
  } finally {
    vi.unstubAllGlobals();
  }
});

test('a 503 from /api/statistics clears the badges instead of throwing', async () => {
  document.body.innerHTML = '<div id="stats-badges"></div>';
  const container = /** @type {HTMLElement} */ (document.getElementById('stats-badges'));
  container.innerHTML = '<span class="stats-badge">stale</span>';
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: false,
    status: 503,
    statusText: 'Service Unavailable',
    json: async () => ({ detail: 'the facility graph is unreachable' }),
  })));
  state.pipelineType = 'graph';

  try {
    await expect(refreshStatsBadges()).resolves.toBeUndefined();
    expect(container.innerHTML).toBe('');
  } finally {
    vi.unstubAllGlobals();
  }
});

test('a rejected /api/statistics fetch clears the badges instead of throwing', async () => {
  document.body.innerHTML = '<div id="stats-badges"></div>';
  const container = /** @type {HTMLElement} */ (document.getElementById('stats-badges'));
  container.innerHTML = '<span class="stats-badge">stale</span>';
  vi.stubGlobal('fetch', vi.fn(async () => {
    throw new TypeError('Failed to fetch');
  }));
  state.pipelineType = 'graph';

  try {
    await expect(refreshStatsBadges()).resolves.toBeUndefined();
    expect(container.innerHTML).toBe('');
  } finally {
    vi.unstubAllGlobals();
  }
});

// ---- The header label map carries the paradigm ----

test('PIPELINE_LABELS has an entry for every pipeline type this view mounts', () => {
  // Without an entry the badge would show the raw key upper-cased.
  for (const { type } of KNOWN_TYPES) {
    expect(PIPELINE_LABELS[type], `label for ${type}`).toBeTruthy();
  }
  expect(PIPELINE_LABELS.graph).toBe('GRAPH');
});
