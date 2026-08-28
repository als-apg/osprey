// @ts-check
/**
 * Unit tests for the graph explorer's renderer (explore-graph.js).
 *
 * The module owns everything between the ontology endpoint and the DOM: the
 * panel shell (title, tool chips, store badge), the SVG class tree, the
 * truncation caution, and the informational pane the empty and unreachable
 * stores share. Run with:
 *   npx vitest run tests/interfaces/channel_finder/explore-graph-renderer.test.mjs
 */

import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest';

import { mountGraph, unmountGraph } from '../../../src/osprey/interfaces/channel_finder/static/js/explore-graph.js';
import { state } from '../../../src/osprey/interfaces/channel_finder/static/js/state.js';

const ONTOLOGY_PATH = '/api/graph/ontology';
const SEM = 'https://narad.example.org/schema/shared_semantics/';

/** The relationship vocabulary the demo corpus seeds. */
const DEMO_RELATIONSHIPS = ['HASBINDING', 'READSSIGNAL', 'SUBCLASSOF', 'TYPE', 'WRITESSIGNAL'];

/**
 * Demo-shaped ontology payload: 19 device classes under one root, exactly as
 * `GET /api/graph/ontology` answers for the seeded demo corpus.
 *
 * @param {Record<string, any>} [overrides] - Payload fields to replace.
 * @returns {Record<string, any>} The response body.
 */
function demoPayload(overrides = {}) {
  /** @type {[string, string[]][]} */
  const spec = [
    ['AcceleratorDevice', []],
    ['Magnet', ['AcceleratorDevice']],
    ['Diagnostic', ['AcceleratorDevice']],
    ['VacuumDevice', ['AcceleratorDevice']],
    ['RFDevice', ['AcceleratorDevice']],
    ['PowerSupply', ['AcceleratorDevice']],
    ['Dipole', ['Magnet']],
    ['Quadrupole', ['Magnet']],
    ['Sextupole', ['Magnet']],
    ['Corrector', ['Magnet']],
    ['SkewQuadrupole', ['Quadrupole']],
    ['HorizontalCorrector', ['Corrector']],
    ['VerticalCorrector', ['Corrector']],
    ['BeamPositionMonitor', ['Diagnostic']],
    ['CurrentMonitor', ['Diagnostic']],
    ['IonPump', ['VacuumDevice']],
    ['VacuumValve', ['VacuumDevice']],
    ['Cavity', ['RFDevice']],
    ['Klystron', ['RFDevice']],
  ];
  return {
    classes: spec.map(([name, parents], i) => ({
      uri: SEM + name,
      name,
      altLabel: [],
      parents: parents.map((p) => SEM + p),
      rollup: (i + 1) * 8,
    })),
    relationship_types: DEMO_RELATIONSHIPS.slice(),
    truncated: false,
    empty: false,
    suggestions: [],
    ...overrides,
  };
}

/**
 * Stub `fetch` with a per-path router.
 *
 * @param {(path: string) => {ok: boolean, status: number, body: any}} route - Reply for a path.
 * @returns {any} The vi mock, for call assertions.
 */
function stubFetch(route) {
  const fn = vi.fn(async (/** @type {any} */ input) => {
    const reply = route(String(input));
    return { ok: reply.ok, status: reply.status, json: async () => reply.body };
  });
  vi.stubGlobal('fetch', fn);
  return fn;
}

/**
 * Stub `fetch` so the ontology endpoint answers 200 with `payload`.
 *
 * @param {Record<string, any>} payload - The response body.
 * @returns {any} The vi mock.
 */
function stubOntology(payload) {
  return stubFetch(() => ({ ok: true, status: 200, body: payload }));
}

/**
 * Count how many times the ontology endpoint was requested.
 *
 * @param {any} fetchMock - The mock returned by `stubFetch`.
 * @returns {number} Request count.
 */
function ontologyCalls(fetchMock) {
  return fetchMock.mock.calls.filter((/** @type {any[]} */ call) =>
    String(call[0]).includes(ONTOLOGY_PATH)).length;
}

/** @returns {HTMLElement} The freshly mounted host element. */
function host() {
  const el = document.getElementById('explore-content');
  if (!el) throw new Error('no host element');
  return /** @type {HTMLElement} */ (el);
}

beforeEach(() => {
  document.body.innerHTML = '<div id="explore-content"></div>';
  state.setGraphInfo(['read_cypher', 'get_schema'], {
    uri: 'bolt://localhost:7687',
    ttl_filename: 'als_corpus.ttl',
  });
});

afterEach(() => {
  unmountGraph();
  state.setGraphInfo([], null);
  vi.unstubAllGlobals();
});

describe('the class tree', () => {
  test('a demo-shaped payload draws every class, the forest, and the panel chrome', async () => {
    stubOntology(demoPayload());

    await mountGraph(host());
    const container = host();

    // One group per class, and the taxonomy is a forest with a single root.
    expect(container.querySelectorAll('.g-node')).toHaveLength(19);
    expect(container.querySelectorAll('.g-node.root')).toHaveLength(1);
    const rootName = container.querySelector('.g-node.root .g-node-name');
    expect(rootName?.textContent).toBe('AcceleratorDevice');

    // 19 classes minus the root leaves 18 drawn SUBCLASSOF links.
    expect(container.querySelectorAll('.g-edge.subclassof')).toHaveLength(18);
    const firstEdge = container.querySelector('.g-edge.subclassof');
    expect(firstEdge?.getAttribute('d')).toMatch(/^M [\d.-]+ [\d.-]+ H /);

    // Rollups render as counts, formatted for the reader.
    const counts = [...container.querySelectorAll('.g-node-count')].map((n) => n.textContent);
    expect(counts).toContain((8).toLocaleString());
    expect(counts).toHaveLength(19);

    // Legend covers the store's relationship vocabulary, each chip tinted.
    const chips = [...container.querySelectorAll('.graph-legend-chip')];
    expect(chips.map((c) => c.textContent)).toEqual(DEMO_RELATIONSHIPS);
    for (const chip of chips) {
      expect(/** @type {HTMLElement} */ (chip).style.getPropertyValue('--c')).toMatch(/^var\(--/);
    }

    // Tool chips come from state, and the store badge names file and URI.
    const tools = [...container.querySelectorAll('.graph-tool-chip')].map((c) => c.textContent);
    expect(tools).toEqual(['read_cypher', 'get_schema']);
    const badge = container.querySelector('.graph-store-badge');
    expect(badge?.textContent).toContain('als_corpus.ttl');
    expect(badge?.textContent).toContain('bolt://localhost:7687');

    // A healthy tree is not an error report.
    expect(container.querySelector('.explore-unknown')).toBeNull();
    expect(container.querySelector('.graph-truncated')).toBeNull();
  });

  test('store-sourced labels are inert, never markup', async () => {
    const payload = demoPayload({
      classes: [{
        uri: `${SEM}Magnet`,
        name: 'Magnet',
        altLabel: ['<img src=x onerror=alert(1)>'],
        parents: [],
        rollup: 4,
      }],
    });
    stubOntology(payload);

    await mountGraph(host());
    const container = host();

    expect(container.querySelectorAll('img')).toHaveLength(0);
    expect(container.innerHTML).not.toContain('<img');
    // The label still reaches the reader — as text, in the node's tooltip.
    const title = container.querySelector('.g-node title');
    expect(title?.textContent).toContain('<img src=x onerror=alert(1)>');
  });

  test('a truncated result carries a caution above the tree', async () => {
    stubOntology(demoPayload({ truncated: true }));

    await mountGraph(host());
    const strip = host().querySelector('.graph-truncated');

    expect(strip).not.toBeNull();
    expect(strip?.textContent?.length).toBeGreaterThan(0);
    // A caution, not a failure: the tree is still drawn.
    expect(host().querySelectorAll('.g-node')).toHaveLength(19);
  });

  test('a store with no seed file is badged by URI alone', async () => {
    state.setGraphInfo(['read_cypher'], { uri: 'bolt://graphdb:7687', ttl_filename: null });
    stubOntology(demoPayload({
      classes: [{ uri: `${SEM}Magnet`, name: 'Magnet', altLabel: [], parents: [], rollup: 4 }],
    }));

    await mountGraph(host());
    const container = host();

    const badge = container.querySelector('.graph-store-badge');
    expect(badge?.textContent).toContain('bolt://graphdb:7687');
    // A missing filename is an absent half of the badge, never a printed word.
    expect(container.innerHTML).not.toContain('null');
    expect(container.innerHTML).not.toContain('undefined');
  });
});

describe('the informational pane', () => {
  test('an empty store offers the seed command, not an error', async () => {
    stubOntology({
      classes: [],
      relationship_types: [],
      truncated: false,
      empty: true,
      suggestions: ['Seed it with `osprey knowledge seed-graph`.'],
    });

    await mountGraph(host());
    const container = host();

    const pane = container.querySelector('.explore-unknown--info');
    expect(pane).not.toBeNull();
    expect(pane?.textContent).toContain('osprey knowledge seed-graph');
    expect(container.querySelector('#graph-retry')).not.toBeNull();
    // Nothing is drawn, and nothing claims to be broken.
    expect(container.querySelectorAll('.g-node')).toHaveLength(0);
    expect(pane?.getAttribute('role')).toBeNull();
  });

  test('an unreachable store shows its detail and remedies, and Retry re-asks', async () => {
    const fetchMock = stubFetch((path) => (
      path.includes(ONTOLOGY_PATH)
        ? {
          ok: false,
          status: 503,
          body: {
            detail: 'Graph store is not reachable at bolt://localhost:7687.',
            error_type: 'service_unavailable',
            suggestions: ['Start the graphdb service.', 'Check services.graphdb.uri.'],
          },
        }
        : { ok: true, status: 200, body: {} }
    ));

    await mountGraph(host());
    const container = host();

    const pane = container.querySelector('.explore-unknown--info');
    expect(pane?.textContent).toContain('Graph store is not reachable at bolt://localhost:7687.');
    expect(pane?.textContent).toContain('Start the graphdb service.');
    expect(pane?.textContent).toContain('Check services.graphdb.uri.');
    expect(ontologyCalls(fetchMock)).toBe(1);

    const retry = /** @type {HTMLElement} */ (container.querySelector('#graph-retry'));
    retry.click();

    await vi.waitFor(() => expect(ontologyCalls(fetchMock)).toBe(2));
  });
});

test('unmounting clears the pane', async () => {
  stubOntology(demoPayload());

  await mountGraph(host());
  expect(host().querySelectorAll('.g-node').length).toBeGreaterThan(0);

  unmountGraph();

  expect(host().innerHTML).toBe('');
});
