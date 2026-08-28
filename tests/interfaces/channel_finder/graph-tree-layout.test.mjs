// @ts-check
/**
 * Unit tests for the graph explorer's class-tree layout (graph-tree-layout.js).
 * Pure logic, no DOM needed:
 *   npx vitest run tests/interfaces/channel_finder/graph-tree-layout.test.mjs
 */

import { test, expect } from 'vitest';

import { layoutForest } from '../../../src/osprey/interfaces/channel_finder/static/js/graph-tree-layout.js';

const SEM = 'https://narad.example.org/schema/shared_semantics/';

/** Stub text measurer: deterministic, no canvas / DOM. */
const measure = (/** @type {string} */ name, /** @type {number} */ rollup) =>
  40 + name.length * 6 + String(rollup).length * 4;

const METRICS = { rowHeight: 30, colWidth: 220, nodeHeight: 24, measure };

/**
 * Demo-shaped 19-class ontology: one root, four depth levels.
 *
 * @returns {import('../../../src/osprey/interfaces/channel_finder/static/js/graph-tree-layout.js').ClassInput[]}
 */
function demoClasses() {
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
  return spec.map(([name, parents], i) => ({
    uri: SEM + name,
    name,
    altLabel: [],
    parents: parents.map((p) => SEM + p),
    rollup: (i + 1) * 3,
  }));
}

/**
 * @param {import('../../../src/osprey/interfaces/channel_finder/static/js/graph-tree-layout.js').ForestLayout} layout
 * @param {string} name
 * @returns {import('../../../src/osprey/interfaces/channel_finder/static/js/graph-tree-layout.js').LayoutNode}
 */
function nodeNamed(layout, name) {
  const found = layout.nodes.find((n) => n.name === name);
  if (!found) throw new Error(`no node named ${name}`);
  return found;
}

test('demo ontology lays out as four depth columns under a single root', () => {
  const layout = layoutForest(demoClasses(), METRICS);

  expect(layout.nodes).toHaveLength(19);
  const depths = new Set(layout.nodes.map((n) => n.depth));
  expect([...depths].sort((a, b) => a - b)).toEqual([0, 1, 2, 3]);

  const root = nodeNamed(layout, 'AcceleratorDevice');
  expect(root.isRoot).toBe(true);
  expect(layout.nodes.filter((n) => n.isRoot)).toHaveLength(1);

  // Depth drives x; every node in a column shares its left edge.
  expect(root.x).toBe(0);
  expect(nodeNamed(layout, 'Magnet').x).toBe(220);
  expect(nodeNamed(layout, 'Corrector').x).toBe(440);
  expect(nodeNamed(layout, 'HorizontalCorrector').x).toBe(660);

  // Widths come from the injected measurer; heights from nodeHeight.
  expect(root.width).toBe(measure('AcceleratorDevice', 3));
  expect(root.height).toBe(24);
});

test('every parent is vertically centred on the span of its children', () => {
  const layout = layoutForest(demoClasses(), METRICS);

  const root = nodeNamed(layout, 'AcceleratorDevice');
  const rootKids = ['Diagnostic', 'Magnet', 'PowerSupply', 'RFDevice', 'VacuumDevice'].map((n) =>
    nodeNamed(layout, n),
  );
  expect(root.y).toBe((rootKids[0].y + rootKids[rootKids.length - 1].y) / 2);

  const corrector = nodeNamed(layout, 'Corrector');
  const horizontal = nodeNamed(layout, 'HorizontalCorrector');
  const vertical = nodeNamed(layout, 'VerticalCorrector');
  expect(corrector.y).toBe((horizontal.y + vertical.y) / 2);

  // Leaves occupy successive rows spaced by rowHeight, starting at 0.
  const leafYs = layout.nodes
    .filter((n) => !layout.edges.some((e) => e.from === n.uri))
    .map((n) => n.y)
    .sort((a, b) => a - b);
  expect(leafYs[0]).toBe(0);
  expect(leafYs[1]).toBe(30);
  expect(new Set(leafYs).size).toBe(leafYs.length);
});

test('sibling ordering is by name and the whole layout is deterministic', () => {
  const first = layoutForest(demoClasses(), METRICS);
  const shuffled = demoClasses().reverse();
  const second = layoutForest(shuffled, METRICS);

  expect(second).toEqual(first);

  const nameByUri = new Map(first.nodes.map((n) => [n.uri, n.name]));
  const rootUri = SEM + 'AcceleratorDevice';
  const childNames = first.edges.filter((e) => e.from === rootUri).map((e) => nameByUri.get(e.to));
  expect(childNames).toEqual(['Diagnostic', 'Magnet', 'PowerSupply', 'RFDevice', 'VacuumDevice']);
});

test('edges are orthogonal elbows from the parent right edge to the child left edge', () => {
  const layout = layoutForest(demoClasses(), METRICS);
  const root = nodeNamed(layout, 'AcceleratorDevice');
  const magnet = nodeNamed(layout, 'Magnet');

  const round2 = (/** @type {number} */ v) => Math.round(v * 100) / 100;
  const edge = layout.edges.find((e) => e.from === root.uri && e.to === magnet.uri);
  expect(edge).toBeDefined();
  const startX = root.x + root.width;
  const midX = round2(startX + (magnet.x - startX) / 2);
  expect(edge?.d).toBe(
    `M ${startX} ${round2(root.y + 12)} H ${midX} V ${round2(magnet.y + 12)} H ${magnet.x}`,
  );
  expect(layout.edges).toHaveLength(18);
});

test('a two-parent class is drawn under the lexicographically first parent uri', () => {
  const classes = [
    { uri: 'urn:b', name: 'Bravo', altLabel: [], parents: [], rollup: 0 },
    { uri: 'urn:a', name: 'Alpha', altLabel: [], parents: [], rollup: 0 },
    { uri: 'urn:c', name: 'Charlie', altLabel: [], parents: ['urn:b', 'urn:a'], rollup: 5 },
  ];

  const layout = layoutForest(classes, METRICS);
  const charlie = nodeNamed(layout, 'Charlie');

  expect(charlie.isRoot).toBe(false);
  expect(charlie.depth).toBe(1);
  expect(charlie.extraParents).toEqual(['urn:b']);
  expect(layout.edges.filter((e) => e.to === 'urn:c')).toEqual([
    expect.objectContaining({ from: 'urn:a', to: 'urn:c' }),
  ]);
});

test('classes whose parents are unknown or absent become stacked roots', () => {
  const classes = [
    { uri: 'urn:z', name: 'Zulu', altLabel: [], parents: ['urn:missing'], rollup: 1 },
    { uri: 'urn:y', name: 'Yankee', altLabel: [], parents: [], rollup: 2 },
    { uri: 'urn:x', name: 'Xray', altLabel: [], rollup: 3 },
  ];

  const layout = layoutForest(classes, METRICS);

  expect(layout.nodes.map((n) => n.name)).toEqual(['Xray', 'Yankee', 'Zulu']);
  expect(layout.nodes.every((n) => n.isRoot && n.depth === 0 && n.x === 0)).toBe(true);
  expect(layout.nodes.map((n) => n.y)).toEqual([0, 30, 60]);
  expect(layout.edges).toEqual([]);
  expect(layout.height).toBe(84);
});

test('a parents cycle terminates and is broken deterministically', () => {
  const classes = [
    { uri: 'urn:a', name: 'Alpha', altLabel: [], parents: ['urn:c'], rollup: 0 },
    { uri: 'urn:b', name: 'Bravo', altLabel: [], parents: ['urn:a'], rollup: 0 },
    { uri: 'urn:c', name: 'Charlie', altLabel: [], parents: ['urn:b'], rollup: 0 },
  ];

  const layout = layoutForest(classes, METRICS);

  expect(layout.nodes).toHaveLength(3);
  const roots = layout.nodes.filter((n) => n.isRoot);
  // The cycle member whose uri sorts first loses its parent link.
  expect(roots.map((n) => n.uri)).toEqual(['urn:a']);
  expect(nodeNamed(layout, 'Alpha').extraParents).toEqual(['urn:c']);
  expect(layout.nodes.map((n) => n.depth).sort()).toEqual([0, 1, 2]);
});

test('a self-parenting class is treated as a root', () => {
  const layout = layoutForest(
    [{ uri: 'urn:self', name: 'Self', altLabel: [], parents: ['urn:self'], rollup: 0 }],
    METRICS,
  );

  expect(layout.nodes).toHaveLength(1);
  expect(layout.nodes[0].isRoot).toBe(true);
  expect(layout.nodes[0].extraParents).toEqual([]);
});

test('an empty class list yields an empty layout', () => {
  const layout = layoutForest([], METRICS);
  expect(layout).toEqual({ nodes: [], edges: [], width: 0, height: 0 });
});

test('a 600-class synthetic forest lays out well under 50 ms', () => {
  /** @type {import('../../../src/osprey/interfaces/channel_finder/static/js/graph-tree-layout.js').ClassInput[]} */
  const classes = [];
  // 24 forests x (1 root + 4 mid classes + 20 leaves) = 600 classes.
  for (let branch = 0; branch < 24; branch += 1) {
    const branchUri = `urn:branch:${branch}`;
    classes.push({ uri: branchUri, name: `Branch${branch}`, altLabel: [], parents: [], rollup: 0 });
    for (let mid = 0; mid < 4; mid += 1) {
      const midUri = `${branchUri}:mid:${mid}`;
      classes.push({
        uri: midUri,
        name: `Branch${branch}Mid${mid}`,
        altLabel: [],
        parents: [branchUri],
        rollup: mid,
      });
      for (let leaf = 0; leaf < 5; leaf += 1) {
        classes.push({
          uri: `${midUri}:leaf:${leaf}`,
          name: `Branch${branch}Mid${mid}Leaf${leaf}`,
          altLabel: [],
          parents: [midUri],
          rollup: leaf,
        });
      }
    }
  }
  expect(classes).toHaveLength(600);

  const started = performance.now();
  const layout = layoutForest(classes, METRICS);
  const elapsed = performance.now() - started;

  expect(layout.nodes).toHaveLength(600);
  expect(layout.edges).toHaveLength(576);
  expect(elapsed).toBeLessThan(50);
});
