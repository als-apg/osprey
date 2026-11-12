// @ts-check
/**
 * The Graph view's model (graph-model.js): what is drawn of the facility
 * graph, and where:
 *   npx vitest run tests/interfaces/okf_panel/graph-model.test.mjs
 */

import { describe, expect, test } from 'vitest';

import {
  count,
  fitTransform,
  HUB_DEGREE,
  indexGraph,
  layout,
  MAX_NODES,
  neighbourhood,
  nodeOfRecord,
  overview,
  relationsOf,
  relationStyle,
  searchNodes,
  zoomAt,
} from '../../../src/osprey/interfaces/okf_panel/static/js/graph-model.js';
import { payload } from './graph-fixture.mjs';

const ALL = () => new Set(indexGraph(payload()).relationCounts.keys());

/**
 * A star: node 0 joined to `leaves` others, and the first leaf to one more.
 * @param {number} leaves
 */
function star(leaves) {
  /** @type {any[]} */
  const nodes = [];
  /** @type {[number, number, string][]} */
  const edges = [];
  for (let i = 0; i <= leaves + 1; i++) {
    nodes.push({ id: 'n' + i, kind: 'channel', label: 'n' + i, type: null, props: {} });
  }
  for (let i = 1; i <= leaves; i++) edges.push([0, i, 'hasBinding']);
  edges.push([1, leaves + 1, 'hasBinding']);
  return indexGraph({ nodes, edges });
}

describe('indexGraph', () => {
  test('counts relations in display order, unknown ones last', () => {
    const graph = indexGraph(payload());
    expect([...graph.relationCounts]).toEqual([
      ['hasBinding', 2],
      ['readsSignal', 1],
      ['writesSignal', 1],
      ['type', 2],
      ['inPlace', 2],
      ['calibrates', 1],
    ]);
    expect(graph.kindCounts.get('device')).toBe(2);
    expect(graph.derived.has('inPlace')).toBe(true);
    expect(graph.classIndex.get('Quadrupole')).toBe(9);
  });

  test('an empty payload indexes to an empty graph', () => {
    const graph = indexGraph(/** @type {any} */ ({}));
    expect(graph.nodes).toEqual([]);
    expect(graph.relationCounts.size).toBe(0);
  });

  test('a relation the stylesheet does not know is styled as other', () => {
    expect(relationStyle('hasBinding')).toBe('hasBinding');
    expect(relationStyle('calibrates')).toBe('other');
  });
});

describe('neighbourhood', () => {
  test('holds the nodes within the asked hops', () => {
    const graph = indexGraph(payload());
    const one = neighbourhood(graph, { focus: 2, hops: 1, relations: ALL() });
    expect(one.nodes.map((node) => node.key).sort()).toEqual([1, 2, 4, 8]);
    const two = neighbourhood(graph, { focus: 2, hops: 2, relations: ALL() });
    expect(two.nodes.map((node) => node.key).sort()).toEqual([1, 2, 3, 4, 6, 8]);
    expect(two.hidden).toBe(0);
  });

  test('follows only the relations that are on', () => {
    const graph = indexGraph(payload());
    const drawn = neighbourhood(graph, { focus: 2, hops: 3, relations: new Set(['hasBinding']) });
    expect(drawn.nodes.map((node) => node.key).sort()).toEqual([2, 4]);
    expect(drawn.edges).toEqual([{ a: 2, b: 4, relation: 'hasBinding', label: 'hasBinding' }]);
  });

  test('stops at a hub that is not the focus, and counts what it left out', () => {
    const graph = star(HUB_DEGREE + 1);
    const relations = new Set(['hasBinding']);
    const far = HUB_DEGREE + 2;
    // From the far leaf: n1 is walked, the hub n0 is reached but not walked.
    const drawn = neighbourhood(graph, { focus: far, hops: 3, relations });
    expect(drawn.nodes.map((node) => node.key).sort((a, b) => Number(a) - Number(b))).toEqual([
      0,
      1,
      far,
    ]);
    expect(drawn.hidden).toBe(HUB_DEGREE);
    expect(drawn.nodes.find((node) => node.key === 0)?.badge).toBe(HUB_DEGREE);

    const opened = neighbourhood(graph, { focus: far, hops: 3, relations, expanded: new Set([0]) });
    expect(opened.hidden).toBe(0);
    expect(opened.nodes).toHaveLength(HUB_DEGREE + 3);
  });

  test('a hub that is the focus is walked', () => {
    const drawn = neighbourhood(star(HUB_DEGREE + 5), {
      focus: 0,
      hops: 1,
      relations: new Set(['hasBinding']),
    });
    expect(drawn.nodes).toHaveLength(HUB_DEGREE + 6);
  });

  test('never holds more than MAX_NODES', () => {
    const drawn = neighbourhood(star(MAX_NODES + 30), {
      focus: 0,
      hops: 1,
      relations: new Set(['hasBinding']),
    });
    expect(drawn.nodes).toHaveLength(MAX_NODES);
    expect(drawn.hidden).toBe(31);
  });
});

describe('overview', () => {
  test('groups devices by class and the rest by kind', () => {
    const drawn = overview(indexGraph(payload()), ALL());
    expect(drawn.nodes.map((node) => node.label)).toEqual([
      'Places · 2',
      'Channels · 2',
      'Signals · 2',
      'BeamPositionMonitor · 1',
      'Quadrupole · 1',
    ]);
    const bpm = drawn.nodes.find((node) => node.key === 'c:8');
    expect(bpm?.members).toEqual([2]);
    expect(bpm?.classNode).toBe(8);
  });

  test('counts the relations between groups onto one edge, and draws no type edge', () => {
    const drawn = overview(indexGraph(payload()), ALL());
    const labels = drawn.edges.map((edge) => `${edge.a} ${edge.label} ${edge.b}`);
    expect(labels).toContain('g:channel readsSignal g:signal');
    expect(labels).toContain('c:9 inPlace g:place');
    expect(drawn.edges.some((edge) => edge.relation === 'type')).toBe(false);
  });

  test('several relations between the same groups are counted', () => {
    const data = payload();
    data.edges.push([5, 6, 'readsSignal']);
    const drawn = overview(indexGraph(data), new Set(['readsSignal']));
    expect(drawn.edges).toEqual([
      expect.objectContaining({ a: 'g:channel', b: 'g:signal', label: 'readsSignal ×2' }),
    ]);
  });
});

describe('layout and the view transform', () => {
  test('one graph always gets one picture', () => {
    const drawn = overview(indexGraph(payload()), ALL());
    const first = new Map();
    const second = new Map();
    layout(drawn, first, { spacing: 120, gravity: 0.25 });
    layout(drawn, second, { spacing: 120, gravity: 0.25 });
    expect([...first]).toEqual([...second]);
    for (const point of first.values()) {
      expect(Number.isFinite(point.x) && Number.isFinite(point.y)).toBe(true);
    }
  });

  test('the pinned node stays at the origin', () => {
    const graph = indexGraph(payload());
    const drawn = neighbourhood(graph, { focus: 2, hops: 2, relations: ALL() });
    const positions = new Map();
    layout(drawn, positions, { spacing: 70, gravity: 0.04, pin: 2 });
    expect(positions.get(2)).toEqual({ x: 0, y: 0 });
  });

  test('a node keeps its place unless the layout is fresh', () => {
    const drawn = {
      nodes: [{ key: 1, label: 'a', kind: /** @type {const} */ ('device'), r: 8 }],
      edges: [],
      hidden: 0,
    };
    const positions = new Map([[1, { x: 40, y: 0 }]]);
    layout(drawn, positions, { spacing: 70, gravity: 0, iterations: 1 });
    expect(positions.get(1)).toEqual({ x: 40, y: 0 });
    layout(drawn, positions, { spacing: 70, gravity: 0, iterations: 1, fresh: true });
    expect(positions.get(1)).not.toEqual({ x: 40, y: 0 });
  });

  test('fitTransform centres the points in the box', () => {
    const tf = fitTransform(
      [
        { x: -100, y: -50 },
        { x: 100, y: 50 },
      ],
      800,
      600
    );
    expect(tf).toEqual({ k: 1.6, x: 400, y: 300 });
    expect(fitTransform([], 800, 600)).toBeNull();
    expect(fitTransform([{ x: 0, y: 0 }], 0, 0)).toBeNull();
  });

  test('zoomAt keeps the point under the cursor and clamps the scale', () => {
    const tf = zoomAt({ x: 0, y: 0, k: 1 }, 2, 100, 50);
    expect(tf).toEqual({ k: 2, x: -100, y: -50 });
    expect(zoomAt({ x: 0, y: 0, k: 1 }, 100, 0, 0).k).toBe(4);
    expect(zoomAt({ x: 0, y: 0, k: 1 }, 0.001, 0, 0).k).toBe(0.15);
  });
});

describe('lookups', () => {
  test('search ranks a label prefix over a label match over an id match', () => {
    const graph = indexGraph(payload());
    expect(searchNodes(graph, 'm:q')).toEqual([5]);
    expect(searchNodes(graph, 'bpm1').slice(0, 2)).toEqual([2, 4]);
    expect(searchNodes(graph, 'Quad')).toEqual([9]);
    expect(searchNodes(graph, 's:cur')).toEqual([7]);
    expect(searchNodes(graph, '  ')).toEqual([]);
    expect(searchNodes(graph, 'm', 3)).toHaveLength(3);
  });

  test('relationsOf groups a node’s relations by direction', () => {
    const graph = indexGraph(payload());
    expect(relationsOf(graph, 4)).toEqual([
      { relation: 'hasBinding', outgoing: false, nodes: [2] },
      { relation: 'readsSignal', outgoing: true, nodes: [6] },
      { relation: 'calibrates', outgoing: false, nodes: [3] },
    ]);
  });

  test('nodeOfRecord finds the node of a facility record', () => {
    const graph = indexGraph(payload());
    expect(nodeOfRecord(graph, 'device', 'M/Q1')).toBe(3);
    expect(nodeOfRecord(graph, 'channel', 'M/Q1')).toBe(-1);
  });

  test('count groups digits the same in every locale', () => {
    expect(count(11373)).toBe('11,373');
  });
});
