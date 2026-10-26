// C-DROP result rect: every hit case's highlight replays against resultRect,
// plus the edges the cases leave implicit.
import { describe, expect, it } from 'vitest';
import { loadCases, runHitResultRect } from './cases.js';
import { planDrop, tileOf } from './drop.js';
import { geometry } from './geometry.js';
import type { TabRects } from './hitTest.js';
import type { Layout } from './layout.js';
import { resultRect } from './resultRect.js';

describe('hit cases (result rect)', () => {
  const loaded = loadCases('hit');

  it('has cases', () => {
    expect(loaded.length).toBeGreaterThan(0);
  });

  it.each(loaded.map((c) => [c.file, c]))('%s', (_file, c) => {
    expect(runHitResultRect(c.data, { resultRect })).toEqual([]);
  });
});

// 50/30/20 row on a 1000 x 600 dock, 30 px bars.
const ROW: Layout = {
  split: 'row',
  items: [
    { tile: 't1', tabs: ['a', 'e'], showing: 'a', size: 0.5 },
    { tile: 't2', tabs: ['b'], showing: 'b', size: 0.3 },
    { tile: 't3', tabs: ['c'], showing: 'c', size: 0.2 },
  ],
};
const DOCK = { x: 0, y: 0, width: 1000, height: 600 };
const BARS: TabRects = {
  t1: {
    barRect: { x: 0, y: 0, width: 500, height: 30 },
    tabRects: [
      { x: 0, y: 0, width: 100, height: 30 },
      { x: 100, y: 0, width: 100, height: 30 },
    ],
  },
  t2: {
    barRect: { x: 500, y: 0, width: 300, height: 30 },
    tabRects: [{ x: 500, y: 0, width: 100, height: 30 }],
  },
  t3: { barRect: { x: 800, y: 0, width: 200, height: 30 }, tabRects: [] },
};

describe('resultRect edges', () => {
  it('puts the marker before the tab at the insertion index', () => {
    const r = resultRect(ROW, DOCK, BARS, { new: 'd' }, { tile: 't1', zone: 'tab', index: 1 });
    expect(r).toEqual({
      rect: { x: 0, y: 0, width: 500, height: 600 },
      marker: { x: 100, y: 0, height: 30 },
    });
  });

  it('appends after the last tab for a centre drop', () => {
    const r = resultRect(ROW, DOCK, BARS, { new: 'd' }, { tile: 't1', zone: 'centre' });
    expect(r?.marker).toEqual({ x: 200, y: 0, height: 30 });
  });

  it('starts the marker at the bar when the bar has no tabs', () => {
    const r = resultRect(ROW, DOCK, BARS, { new: 'd' }, { tile: 't3', zone: 'centre' });
    expect(r?.marker).toEqual({ x: 800, y: 0, height: 30 });
  });

  it('has no marker for a tile the dock did not measure', () => {
    const r = resultRect(ROW, DOCK, {}, { new: 'd' }, { tile: 't2', zone: 'centre' });
    expect(r).toEqual({ rect: { x: 500, y: 0, width: 300, height: 600 }, marker: null });
  });

  it('is null for an unknown target tile', () => {
    expect(resultRect(ROW, DOCK, BARS, { new: 'd' }, { tile: 'nope', zone: 'centre' })).toBeNull();
    expect(resultRect(ROW, DOCK, BARS, { new: 'd' }, { tile: 'nope', edge: 'left' })).toBeNull();
  });

  it("highlights the hovered tile's half for an edge drop, bar included", () => {
    expect(resultRect(ROW, DOCK, BARS, { tab: 'e' }, { tile: 't1', edge: 'bottom' })).toEqual({
      rect: { x: 0, y: 300, width: 500, height: 300 },
      marker: null,
    });
    expect(resultRect(ROW, DOCK, BARS, { new: 'd' }, { tile: 't2', edge: 'right' })).toEqual({
      rect: { x: 650, y: 0, width: 150, height: 600 },
      marker: null,
    });
  });

  it('stays inside the hovered tile when its neighbour is the dragged tile', () => {
    const r = resultRect(ROW, DOCK, BARS, { tile: 't3' }, { tile: 't2', edge: 'right' });
    expect(r).toEqual({ rect: { x: 650, y: 0, width: 150, height: 600 }, marker: null });
  });

  it("equals the landed tile when the dragged item's tile stays, odd spans included", () => {
    const odd = { x: 3, y: 7, width: 1001, height: 601 };
    const dock = { px: { width: 1001, height: 601 }, minPx: 40 };
    for (const edge of ['left', 'right', 'top', 'bottom'] as const) {
      for (const [source, key] of [
        [{ tab: 'e' }, 'e'],
        [{ new: 'd' }, 'd'],
      ] as const) {
        const target = { tile: 't1', edge };
        const after = planDrop(ROW, source, target, dock).layout!;
        const landed = tileOf(after, key)!.tile;
        expect(resultRect(ROW, odd, BARS, source, target)?.rect).toEqual(
          geometry(after, odd, 40).tiles[landed]
        );
      }
    }
  });

  it('is null for a self-drop planDrop refuses', () => {
    expect(resultRect(ROW, DOCK, BARS, { tile: 't2' }, { tile: 't2', edge: 'left' })).toBeNull();
  });

  it('is null for border targets, which no pointer zone makes', () => {
    expect(
      resultRect(ROW, DOCK, BARS, { new: 'd' }, { border: { tile: 't1', side: 'right' } })
    ).toBeNull();
  });

  it('is the new full-span strip for outer targets, as the drop lays it out', () => {
    const dock = { px: { width: 1000, height: 600 }, minPx: 40 };
    for (const side of ['left', 'right', 'top', 'bottom'] as const) {
      for (const [source, key] of [
        [{ tab: 'e' }, 'e'],
        [{ new: 'd' }, 'd'],
      ] as const) {
        const after = planDrop(ROW, source, { outer: side }, dock).layout!;
        const landed = tileOf(after, key)!.tile;
        expect(resultRect(ROW, DOCK, BARS, source, { outer: side })).toEqual({
          rect: geometry(after, DOCK, 40).tiles[landed],
          marker: null,
        });
      }
    }
    expect(resultRect(ROW, DOCK, BARS, { new: 'd' }, { outer: 'bottom' })?.rect).toEqual({
      x: 0,
      y: 400,
      width: 1000,
      height: 200,
    });
  });

  it('is null for an outer drop planDrop refuses', () => {
    const one: Layout = { tile: 't1', tabs: ['a'], showing: 'a' };
    expect(resultRect(one, DOCK, {}, { tile: 't1' }, { outer: 'left' })).toBeNull();
  });

  it('honours the dock offset', () => {
    const offset = { x: 10, y: 40, width: 1000, height: 600 };
    const r = resultRect(ROW, offset, BARS, { new: 'd' }, { tile: 't1', edge: 'left' });
    expect(r?.rect).toEqual({ x: 10, y: 40, width: 250, height: 600 });
  });

  it('does not mutate its inputs', () => {
    const layout = structuredClone(ROW);
    const bars = structuredClone(BARS);
    resultRect(layout, DOCK, bars, { tab: 'e' }, { tile: 't2', edge: 'right' });
    resultRect(layout, DOCK, bars, { new: 'd' }, { tile: 't1', zone: 'centre' });
    expect(layout).toEqual(ROW);
    expect(bars).toEqual(BARS);
  });
});
