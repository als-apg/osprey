// C-DROP hit precedence: every hit case's target selection replays against
// hitTest, plus the edges the cases leave implicit.
import { describe, expect, it } from 'vitest';
import { loadCases, runHitTarget } from './cases.js';
import { hitTest } from './hitTest.js';
import type { TabRects } from './hitTest.js';
import type { Layout } from './layout.js';

describe('hit cases (target selection)', () => {
  const loaded = loadCases('hit');

  it('has cases', () => {
    expect(loaded.length).toBeGreaterThan(0);
  });

  it.each(loaded.map((c) => [c.file, c]))('%s', (_file, c) => {
    expect(runHitTarget(c.data, { hitTest })).toEqual([]);
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
  t3: {
    barRect: { x: 800, y: 0, width: 200, height: 30 },
    tabRects: [{ x: 800, y: 0, width: 100, height: 30 }],
  },
};

describe('hitTest', () => {
  it('a tab on its own multi-tab bar reorders (not a self-drop)', () => {
    expect(hitTest(ROW, DOCK, BARS, { x: 160, y: 10 }, { tab: 'a' })).toEqual({
      tile: 't1',
      zone: 'tab',
      index: 2,
    });
  });

  it('a tab of a multi-tab tile may split off its own edge', () => {
    expect(hitTest(ROW, DOCK, BARS, { x: 450, y: 300 }, { tab: 'e' })).toEqual({
      tile: 't1',
      edge: 'right',
    });
  });

  it('an already open {new} key whose tile it is alone in self-drops like a tab', () => {
    expect(hitTest(ROW, DOCK, BARS, { x: 650, y: 300 }, { new: 'b' })).toBeNull();
  });

  it('a sole-tab source self-drops on its own tile, but not on its neighbour next to it', () => {
    expect(hitTest(ROW, DOCK, BARS, { x: 795, y: 300 }, { tab: 'b' })).toBeNull();
    expect(hitTest(ROW, DOCK, BARS, { x: 805, y: 300 }, { tab: 'b' })).toEqual({
      tile: 't3',
      edge: 'left',
    });
  });

  it('a tile dropped on another tile is a target', () => {
    expect(hitTest(ROW, DOCK, BARS, { x: 250, y: 300 }, { tile: 't2' })).toEqual({
      tile: 't1',
      zone: 'centre',
    });
  });

  it('a bar of a tile not in the layout is ignored', () => {
    const bars: TabRects = {
      ...BARS,
      gone: { barRect: { x: 0, y: 300, width: 500, height: 30 }, tabRects: [] },
    };
    expect(hitTest(ROW, DOCK, bars, { x: 250, y: 310 }, { new: 'd' })).toEqual({
      tile: 't1',
      zone: 'centre',
    });
  });

  it('the left of the dock rect is off dock', () => {
    expect(hitTest(ROW, DOCK, BARS, { x: -1, y: 300 }, { new: 'd' })).toBeNull();
  });

  it('has no border zone: a split line is two tile edges, the dock sides are strips', () => {
    expect(hitTest(ROW, DOCK, BARS, { x: 499, y: 300 }, { new: 'd' })).toEqual({
      tile: 't1',
      edge: 'right',
    });
    expect(hitTest(ROW, DOCK, BARS, { x: 500, y: 300 }, { new: 'd' })).toEqual({
      tile: 't2',
      edge: 'left',
    });
    expect(hitTest(ROW, DOCK, BARS, { x: 0, y: 300 }, { new: 'd' })).toEqual({ outer: 'left' });
    expect(hitTest(ROW, DOCK, BARS, { x: 250, y: 599 }, { new: 'd' })).toEqual({
      outer: 'bottom',
    });
  });

  it('measures the dock-edge strips STRIP_PX deep, the top one under the header row', () => {
    const at = (x: number, y: number) => hitTest(ROW, DOCK, BARS, { x, y }, { new: 'd' });
    expect(at(47, 300)).toEqual({ outer: 'left' });
    expect(at(48, 300)).toEqual({ tile: 't1', edge: 'left' });
    expect(at(953, 300)).toEqual({ outer: 'right' });
    expect(at(952, 300)).toEqual({ tile: 't3', edge: 'right' });
    expect(at(250, 553)).toEqual({ outer: 'bottom' });
    expect(at(250, 552)).toEqual({ tile: 't1', edge: 'bottom' });
    expect(at(250, 29)).toEqual({ tile: 't1', zone: 'tab', index: 2 });
    expect(at(250, 30)).toEqual({ outer: 'top' });
    expect(at(250, 77)).toEqual({ outer: 'top' });
    expect(at(250, 78)).toEqual({ tile: 't1', edge: 'top' });
  });

  it('has a top strip only in the top row', () => {
    const column: Layout = {
      split: 'column',
      items: [
        { tile: 't1', tabs: ['a'], showing: 'a', size: 0.5 },
        { tile: 't2', tabs: ['b'], showing: 'b', size: 0.5 },
      ],
    };
    const bars: TabRects = {
      t1: { barRect: { x: 0, y: 0, width: 1000, height: 30 }, tabRects: [] },
      t2: { barRect: { x: 0, y: 300, width: 1000, height: 30 }, tabRects: [] },
    };
    expect(hitTest(column, DOCK, bars, { x: 500, y: 335 }, { new: 'd' })).toEqual({
      tile: 't2',
      edge: 'top',
    });
  });

  it('gives a neighbour no shared edge: its band facing the source tile splits it', () => {
    const row: Layout = {
      split: 'row',
      items: [
        { tile: 't1', tabs: ['a', 'b'], showing: 'a', size: 0.5 },
        { tile: 't2', tabs: ['c'], showing: 'c', size: 0.5 },
      ],
    };
    expect(hitTest(row, DOCK, {}, { x: 550, y: 300 }, { tab: 'b' })).toEqual({
      tile: 't2',
      edge: 'left',
    });
  });

  it('the only tile moving to a strip is a self-drop', () => {
    const one: Layout = { tile: 't1', tabs: ['a'], showing: 'a' };
    expect(hitTest(one, DOCK, {}, { x: 10, y: 300 }, { tile: 't1' })).toBeNull();
    expect(hitTest(one, DOCK, {}, { x: 10, y: 300 }, { tab: 'a' })).toBeNull();
    expect(hitTest(one, DOCK, {}, { x: 10, y: 300 }, { new: 'b' })).toEqual({ outer: 'left' });
  });

  it('measures edge depth as max(80, 31 % of the content), the bar excluded', () => {
    // t1 content: 500 x 570 from y 30 -> left/right 155 deep, top/bottom 176.7 deep.
    const at = (x: number, y: number) => hitTest(ROW, DOCK, BARS, { x, y }, { new: 'd' });
    expect(at(154, 300)).toEqual({ tile: 't1', edge: 'left' });
    expect(at(155, 300)).toEqual({ tile: 't1', zone: 'centre' });
    expect(at(346, 300)).toEqual({ tile: 't1', edge: 'right' });
    expect(at(250, 206)).toEqual({ tile: 't1', edge: 'top' });
    expect(at(250, 207)).toEqual({ tile: 't1', zone: 'centre' });
    expect(at(250, 424)).toEqual({ tile: 't1', edge: 'bottom' });
    // t3 is 200 wide: the 80 px floor wins over 31 % (62).
    expect(at(879, 300)).toEqual({ tile: 't3', edge: 'left' });
    expect(at(880, 300)).toEqual({ tile: 't3', zone: 'centre' });
  });

  it('lets left and right win at the corners', () => {
    const at = (x: number, y: number) => hitTest(ROW, DOCK, BARS, { x, y }, { new: 'd' });
    expect(at(60, 90)).toEqual({ tile: 't1', edge: 'left' });
    expect(at(440, 540)).toEqual({ tile: 't1', edge: 'right' });
  });

  it('clamps the edge depth to half the content: a 120 px tile is all left and right', () => {
    const narrow: Layout = {
      split: 'row',
      items: [
        { tile: 't1', tabs: ['a'], showing: 'a', size: 0.88 },
        { tile: 't2', tabs: ['b'], showing: 'b', size: 0.12 },
      ],
    };
    const at = (x: number, y: number) => hitTest(narrow, DOCK, {}, { x, y }, { new: 'd' });
    expect(at(939, 100)).toEqual({ tile: 't2', edge: 'left' });
    expect(at(941, 500)).toEqual({ tile: 't2', edge: 'right' });
  });

  it('gives an axis too short to halve into two 40 px tiles no edges', () => {
    const tight: Layout = {
      split: 'column',
      items: [
        { tile: 't1', tabs: ['a'], showing: 'a', size: 0.8 },
        { tile: 't2', tabs: ['b'], showing: 'b', size: 0.2 },
      ],
    };
    const dock = { x: 0, y: 0, width: 1000, height: 390 };
    // t2 is 78 px tall: no top/bottom zones, its left/right zones still split it.
    expect(hitTest(tight, dock, {}, { x: 500, y: 315 }, { new: 'd' })).toEqual({
      tile: 't2',
      zone: 'centre',
    });
    expect(hitTest(tight, dock, {}, { x: 60, y: 330 }, { new: 'd' })).toEqual({
      tile: 't2',
      edge: 'left',
    });
  });

  it('makes the whole header row one target: right of the tabs appends a tab', () => {
    const narrowBars: TabRects = {
      ...BARS,
      t2: { ...BARS.t2!, barRect: { x: 500, y: 0, width: 120, height: 30 } },
    };
    expect(hitTest(ROW, DOCK, narrowBars, { x: 790, y: 29 }, { new: 'd' })).toEqual({
      tile: 't2',
      zone: 'tab',
      index: 1,
    });
    expect(hitTest(ROW, DOCK, narrowBars, { x: 790, y: 29 }, { tile: 't2' })).toBeNull();
  });

  it('does not mutate its inputs', () => {
    const layout = structuredClone(ROW);
    const bars = structuredClone(BARS);
    hitTest(layout, DOCK, bars, { x: 495, y: 300 }, { tile: 't2' });
    expect(layout).toEqual(ROW);
    expect(bars).toEqual(BARS);
  });
});
