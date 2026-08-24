// @ts-check
/**
 * Dispatch contract for the Explore view (explore.js).
 *
 * Every known pipeline type must reach its own renderer, the graph paradigm
 * must reach its own static "no browser" pane, and a pipeline type this view
 * does not know about must land on the explicit "unknown" pane — never on the
 * in-context renderer as a silent catch-all.
 *   npx vitest run tests/interfaces/channel_finder/explore-graph.test.mjs
 *
 * Runs under happy-dom (vitest.config.js). The three renderer modules are
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
];

const ALL_MOUNTS = KNOWN_TYPES.map(t => t.mount);

/**
 * Put a pipeline type on the shared state and mount Explore into a fresh
 * document, returning the container.
 * @param {string|null} pipelineType
 * @returns {HTMLElement}
 */
function mountWith(pipelineType) {
  state.pipelineType = pipelineType;
  state.pipelineMetadata = {};
  state.dbPath = null;
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

/**
 * The graph paradigm's static pane — same pane style, expected posture.
 * @returns {HTMLElement|null}
 */
const graphPane = () => document.querySelector('[data-pipeline="graph"]');

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

// ---- Graph paradigm: a static pane, not a renderer ----

test('the graph paradigm mounts its static pane and no renderer', () => {
  mountWith('graph');

  const pane = graphPane();
  expect(pane, 'graph pane is mounted').not.toBeNull();
  expect(pane?.textContent).toContain(
    'This paradigm has no browser \u2014 channels are resolved by querying the facility graph'
  );

  // No renderer module ran: the graph paradigm ships no browser at all.
  for (const mount of ALL_MOUNTS) expect(mount).not.toHaveBeenCalled();
  // And it is not the unknown-pipeline notice: graph is a configuration the
  // server accepts, so nothing here may read as "the server rejected this".
  expect(unknownPane(), 'not the unknown-pipeline pane').toBeNull();
  expect(document.body.textContent).not.toContain('Unknown pipeline');
  // Including the tint: the pane wears the informational modifier, which is
  // what keeps a correctly configured deployment from looking like an error.
  expect(pane?.classList.contains('explore-unknown--info')).toBe(true);
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

test('unmounting the graph paradigm clears the pane and calls no renderer unmount', () => {
  mountWith('graph');
  expect(graphPane()).not.toBeNull();

  unmountExplore();

  expect(graphPane(), 'pane is cleared on unmount').toBeNull();
  for (const row of KNOWN_TYPES) expect(row.unmount).not.toHaveBeenCalled();
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

test('the stats badges make no request in graph mode and stay empty', async () => {
  document.body.innerHTML = '<div id="stats-badges"></div>';
  const container = /** @type {HTMLElement} */ (document.getElementById('stats-badges'));
  container.innerHTML = '<span class="stats-badge">stale</span>';
  const fetchSpy = vi.fn();
  vi.stubGlobal('fetch', fetchSpy);
  state.pipelineType = 'graph';

  try {
    await refreshStatsBadges();
    // /api/statistics is a documented 501 in graph mode — never asked for.
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(container.innerHTML).toBe('');
  } finally {
    vi.unstubAllGlobals();
  }
});

test('a 501 from /api/statistics clears the badges instead of throwing', async () => {
  document.body.innerHTML = '<div id="stats-badges"></div>';
  const container = /** @type {HTMLElement} */ (document.getElementById('stats-badges'));
  container.innerHTML = '<span class="stats-badge">stale</span>';
  vi.stubGlobal('fetch', vi.fn(async () => ({
    ok: false,
    status: 501,
    statusText: 'Not Implemented',
    json: async () => ({ detail: 'statistics are not available for the graph paradigm' }),
  })));
  state.pipelineType = 'in_context';

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
