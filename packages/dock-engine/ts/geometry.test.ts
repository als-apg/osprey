// geometry: tile rects and interior borders of a C-LAYOUT tree, laid out with
// dockview-core 7.0.2's splitview arithmetic (round, clamp to the minimum,
// hand the difference out last child first), plus the px -> fraction
// conversion shared by the serializer and the `resized` sync.
import { describe, expect, it } from 'vitest';
import { fractionsOf, geometry, itemFractions, layoutSizes, minimumSize } from './geometry.js';
import type { Layout, Split, Tile } from './layout.js';

function tile(id: string, size?: number): Tile {
  const t: Tile = { tile: id, tabs: [id.toLowerCase()], showing: id.toLowerCase() };
  if (size !== undefined) t.size = size;
  return t;
}

function split(dir: 'row' | 'column', items: Layout[], size?: number): Split {
  return size === undefined ? { split: dir, items } : { split: dir, items, size };
}

const DOCK = { x: 10, y: 20, width: 1000, height: 600 };
const MIN = 40;

describe('geometry: tile rects', () => {
  it('a root tile fills the dock rect', () => {
    const g = geometry(tile('A'), DOCK, MIN);
    expect(g.tiles).toEqual({ A: DOCK });
    expect(g.borders).toEqual([]);
    expect(g.feasible).toBe(true);
  });

  it('lays out the 50/30/20 start A|B|C along the row', () => {
    const g = geometry(split('row', [tile('A', 0.5), tile('B', 0.3), tile('C', 0.2)]), DOCK, MIN);
    expect(g.tiles).toEqual({
      A: { x: 10, y: 20, width: 500, height: 600 },
      B: { x: 510, y: 20, width: 300, height: 600 },
      C: { x: 810, y: 20, width: 200, height: 600 },
    });
  });

  it('stacks a column top to bottom', () => {
    const g = geometry(split('column', [tile('A', 0.25), tile('B', 0.75)]), DOCK, MIN);
    expect(g.tiles.A).toEqual({ x: 10, y: 20, width: 1000, height: 150 });
    expect(g.tiles.B).toEqual({ x: 10, y: 170, width: 1000, height: 450 });
  });

  it('shares missing sizes equally in the remainder', () => {
    const g = geometry(split('row', [tile('A', 0.5), tile('B'), tile('C')]), DOCK, MIN);
    expect(g.tiles.A!.width).toBe(500);
    expect(g.tiles.B!.width).toBe(250);
    expect(g.tiles.C!.width).toBe(250);
  });

  it('rounds like dockview: the last child takes the rounding remainder', () => {
    const g = geometry(split('row', [tile('A'), tile('B'), tile('C')]), DOCK, MIN);
    expect([g.tiles.A!.width, g.tiles.B!.width, g.tiles.C!.width]).toEqual([333, 333, 334]);
  });

  it('nests rects through alternating splits and tiles the dock exactly', () => {
    const layout = split('row', [
      split('column', [tile('A'), tile('B')], 0.5),
      split('column', [tile('C', 0.3), tile('D', 0.7)], 0.5),
    ]);
    const g = geometry(layout, DOCK, MIN);
    expect(g.tiles).toEqual({
      A: { x: 10, y: 20, width: 500, height: 300 },
      B: { x: 10, y: 320, width: 500, height: 300 },
      C: { x: 510, y: 20, width: 500, height: 180 },
      D: { x: 510, y: 200, width: 500, height: 420 },
    });
    const area = Object.values(g.tiles).reduce((sum, r) => sum + r.width * r.height, 0);
    expect(area).toBe(DOCK.width * DOCK.height);
  });

  it('handles a deep alternation without a depth cap', () => {
    let layout: Layout = tile('T0');
    for (let i = 1; i <= 8; i++) {
      layout = split(i % 2 ? 'row' : 'column', [layout, tile(`T${i}`)]);
    }
    const g = geometry(layout, { x: 0, y: 0, width: 4000, height: 4000 }, MIN);
    expect(Object.keys(g.tiles)).toHaveLength(9);
    expect(g.feasible).toBe(true);
    const area = Object.values(g.tiles).reduce((sum, r) => sum + r.width * r.height, 0);
    expect(area).toBe(4000 * 4000);
  });
});

describe('geometry: the 40 px clamp', () => {
  it('a small last tile gets 40 px and the previous sibling gives the difference', () => {
    const g = geometry(split('row', [tile('A', 0.97), tile('B', 0.03)]), DOCK, MIN);
    expect(g.tiles.B!.width).toBe(40);
    expect(g.tiles.A!.width).toBe(960);
    expect(g.tiles.B!.x).toBe(970);
  });

  it('a small first tile gets 40 px and the last sibling gives the difference', () => {
    const g = geometry(split('row', [tile('A', 0.03), tile('B', 0.47), tile('C', 0.5)]), DOCK, MIN);
    expect(g.tiles.A!.width).toBe(40);
    expect(g.tiles.B!.width).toBe(470);
    expect(g.tiles.C!.width).toBe(490);
  });

  it('the difference walks back from the last child when the last is at its minimum', () => {
    const g = geometry(split('row', [tile('A', 0.5), tile('B', 0.47), tile('C', 0.03)]), DOCK, MIN);
    expect([g.tiles.A!.width, g.tiles.B!.width, g.tiles.C!.width]).toEqual([500, 460, 40]);
  });

  it('a branch is clamped to its subtree minimum, not to 40 px', () => {
    // The column on the right holds a row of three tiles: it needs 120 px wide.
    const layout = split('row', [
      tile('A', 0.95),
      split('column', [tile('B'), split('row', [tile('C'), tile('D'), tile('E')])], 0.05),
    ]);
    const g = geometry(layout, DOCK, MIN);
    expect(g.tiles.A!.width).toBe(880);
    expect(g.tiles.B!.width).toBe(120);
    expect([g.tiles.C!.width, g.tiles.D!.width, g.tiles.E!.width]).toEqual([40, 40, 40]);
    expect(g.feasible).toBe(true);
  });

  it('reports infeasible when the dock cannot give every tile its minimum', () => {
    expect(
      geometry(split('row', [tile('A'), tile('B')]), { x: 0, y: 0, width: 70, height: 600 }, MIN)
        .feasible
    ).toBe(false);
    expect(
      geometry(
        split('column', [tile('A'), tile('B'), tile('C')]),
        { x: 0, y: 0, width: 600, height: 100 },
        MIN
      ).feasible
    ).toBe(false);
    expect(
      geometry(split('row', [tile('A'), tile('B')]), { x: 0, y: 0, width: 80, height: 40 }, MIN)
        .feasible
    ).toBe(true);
  });

  it('keeps tiles at their minimums (overflowing) when infeasible, as dockview does', () => {
    const g = geometry(
      split('row', [tile('A'), tile('B')]),
      { x: 0, y: 0, width: 70, height: 600 },
      MIN
    );
    expect(g.tiles.A!.width).toBe(40);
    expect(g.tiles.B!.width).toBe(40);
  });
});

describe('geometry: interior borders', () => {
  it('a row of three has two vertical borders, each naming its two siblings', () => {
    const g = geometry(split('row', [tile('A', 0.5), tile('B', 0.3), tile('C', 0.2)]), DOCK, MIN);
    expect(g.borders).toEqual([
      {
        split: 'row',
        path: [],
        index: 1,
        before: ['A'],
        after: ['B'],
        line: { x1: 510, y1: 20, x2: 510, y2: 620 },
      },
      {
        split: 'row',
        path: [],
        index: 2,
        before: ['B'],
        after: ['C'],
        line: { x1: 810, y1: 20, x2: 810, y2: 620 },
      },
    ]);
  });

  it('a border between two branches separates their whole subtrees', () => {
    const layout = split('row', [
      split('column', [tile('A'), tile('B')]),
      split('column', [tile('C'), tile('D')]),
    ]);
    const g = geometry(layout, DOCK, MIN);
    expect(g.borders).toEqual([
      {
        split: 'row',
        path: [],
        index: 1,
        before: ['A', 'B'],
        after: ['C', 'D'],
        line: { x1: 510, y1: 20, x2: 510, y2: 620 },
      },
      {
        split: 'column',
        path: [0],
        index: 1,
        before: ['A'],
        after: ['B'],
        line: { x1: 10, y1: 320, x2: 510, y2: 320 },
      },
      {
        split: 'column',
        path: [1],
        index: 1,
        before: ['C'],
        after: ['D'],
        line: { x1: 510, y1: 320, x2: 1010, y2: 320 },
      },
    ]);
  });

  it('borders follow the clamped sizes, not the raw fractions', () => {
    const g = geometry(split('row', [tile('A', 0.97), tile('B', 0.03)]), DOCK, MIN);
    expect(g.borders[0]!.line.x1).toBe(970);
  });

  it('the outer edges of the dock are not borders', () => {
    const g = geometry(split('column', [tile('A'), tile('B')]), DOCK, MIN);
    expect(g.borders).toHaveLength(1);
    expect(g.borders[0]!.line).toEqual({ x1: 10, y1: 320, x2: 1010, y2: 320 });
  });
});

describe('minimumSize', () => {
  it('sums along a split axis and takes the largest across it', () => {
    const layout = split('row', [
      tile('A'),
      split('column', [tile('B'), split('row', [tile('C'), tile('D')])]),
    ]);
    expect(minimumSize(layout, 'width', MIN)).toBe(120);
    expect(minimumSize(layout, 'height', MIN)).toBe(80);
    expect(minimumSize(tile('A'), 'width', MIN)).toBe(40);
  });
});

describe('layoutSizes and itemFractions', () => {
  it('reproduces splitview round + clamp + last-first distribution', () => {
    expect(layoutSizes([0.5, 0.3, 0.2], 1000, [40, 40, 40])).toEqual([500, 300, 200]);
    expect(layoutSizes([0.98, 0.01, 0.01], 1000, [40, 40, 40])).toEqual([920, 40, 40]);
    expect(layoutSizes([0.5, 0.5], 70, [40, 40])).toEqual([40, 40]);
  });

  it('reads missing sizes as an equal share of the remainder and renormalises', () => {
    expect(itemFractions([{ size: 0.5 }, {}, {}])).toEqual([0.5, 0.25, 0.25]);
    expect(itemFractions([{}, {}])).toEqual([0.5, 0.5]);
    const scaled = itemFractions([{ size: 2 }, { size: 2 }]);
    expect(scaled).toEqual([0.5, 0.5]);
  });
});

describe('fractionsOf (px -> fraction)', () => {
  it('converts px sizes to fractions that sum to exactly 1', () => {
    expect(fractionsOf([500, 300, 200])).toEqual([0.5, 0.3, 0.2]);
    const thirds = fractionsOf([333, 333, 334]);
    expect(thirds.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    for (const f of thirds) expect(f).toBeCloseTo(1 / 3, 2);
  });

  it('gives equal shares for an all-zero split and nothing for none', () => {
    expect(fractionsOf([0, 0])).toEqual([0.5, 0.5]);
    expect(fractionsOf([])).toEqual([]);
  });

  it('round-trips through geometry within 2 px', () => {
    const layout = split('row', [tile('A', 0.37), tile('B', 0.21), tile('C', 0.42)]);
    const g = geometry(layout, { x: 0, y: 0, width: 1237, height: 400 }, MIN);
    const px = ['A', 'B', 'C'].map((id) => g.tiles[id]!.width);
    const back = fractionsOf(px);
    const again = geometry(
      split(
        'row',
        back.map((f, i) => tile(['A', 'B', 'C'][i]!, f))
      ),
      { x: 0, y: 0, width: 1237, height: 400 },
      MIN
    );
    for (const id of ['A', 'B', 'C']) {
      expect(Math.abs(again.tiles[id]!.width - g.tiles[id]!.width)).toBeLessThanOrEqual(2);
      expect(Math.abs(again.tiles[id]!.x - g.tiles[id]!.x)).toBeLessThanOrEqual(2);
    }
  });
});
