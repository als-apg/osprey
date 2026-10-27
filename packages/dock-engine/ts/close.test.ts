// C-DROP: the close family and reopen (rules 5 and 8) replay every close and
// reopen case, leave valid layouts and never touch their input.
import { describe, expect, it } from 'vitest';
import { diffLayout, loadCases, runOpCase } from './cases.js';
import type { OpCase } from './cases.js';
import { closeAll, closeOthers, closeRight, closeTab, closeTile, reopen } from './close.js';
import type { CloseContext, TileRecord } from './close.js';
import { validateLayout } from './layout.js';
import type { Layout } from './layout.js';

const engine = { closeTab, closeTile, closeOthers, closeRight, closeAll, reopen };

describe('close cases', () => {
  const loaded = loadCases('close');
  it('has cases', () => expect(loaded.length).toBeGreaterThan(0));
  it.each(loaded.map((c) => [c.file, c]))('%s', (_file, c) => {
    expect(runOpCase(c.data as OpCase, engine)).toEqual([]);
  });
});

describe('reopen cases', () => {
  const loaded = loadCases('reopen');
  it('has cases', () => expect(loaded.length).toBeGreaterThan(0));
  it.each(loaded.map((c) => [c.file, c]))('%s', (_file, c) => {
    expect(runOpCase(c.data as OpCase, engine)).toEqual([]);
  });
});

const abc = (): Layout => ({
  split: 'row',
  items: [
    { tile: 't1', tabs: ['a'], showing: 'a', size: 0.5 },
    { tile: 't2', tabs: ['b', 'e', 'f'], showing: 'e', size: 0.3 },
    { tile: 't3', tabs: ['c'], showing: 'c', size: 0.2 },
  ],
});

const dock = { px: { width: 1000, height: 600 }, minPx: 40 };
const ctx = (extra: Partial<CloseContext> = {}): CloseContext => ({
  focusedTile: 't3',
  dock,
  ...extra,
});

describe('closes', () => {
  it('never mutate their input and leave valid layouts', () => {
    const ops = [
      (l: Layout) => closeTab(l, 'e'),
      (l: Layout) => closeTab(l, 'a'),
      (l: Layout) => closeTile(l, 't2'),
      (l: Layout) => closeOthers(l, 'b'),
      (l: Layout) => closeRight(l, 'b'),
      (l: Layout) => closeAll(l),
    ];
    for (const op of ops) {
      const before = abc();
      const copy = structuredClone(before);
      const out = op(before);
      expect(before).toEqual(copy);
      expect(out.changed).toBe(true);
      if (out.layout !== null) expect(validateLayout(out.layout)).toEqual({ valid: true });
    }
  });

  it('a group close that closes nothing changes nothing and records nothing', () => {
    expect(closeOthers(abc(), 'a')).toEqual({ layout: abc(), changed: false, record: null });
    expect(closeRight(abc(), 'f')).toEqual({ layout: abc(), changed: false, record: null });
    expect(closeAll(null)).toEqual({ layout: null, changed: false, record: null });
  });

  it('an unknown tile or key changes nothing', () => {
    expect(closeTile(abc(), 't9')).toEqual({ layout: abc(), changed: false, record: null });
    expect(closeOthers(abc(), 'zz').changed).toBe(false);
    expect(closeRight(null, 'zz').changed).toBe(false);
  });

  it('closing a showing tab to the right shows the kept key', () => {
    const out = closeRight(abc(), 'b');
    expect(out.layout).toEqual({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['a'], showing: 'a', size: 0.5 },
        { tile: 't2', tabs: ['b'], showing: 'b', size: 0.3 },
        { tile: 't3', tabs: ['c'], showing: 'c', size: 0.2 },
      ],
    });
  });
});

describe('close then reopen', () => {
  it('round-trips a tab, a tile, a group and the whole dock', () => {
    const closes = [
      closeTab(abc(), 'e'),
      closeTab(abc(), 'f'),
      closeTile(abc(), 't2'),
      closeTile(abc(), 't3'),
      closeOthers(abc(), 'e'),
      closeAll(abc()),
    ];
    for (const out of closes) {
      expect(out.record).toBeTruthy();
      const back = reopen(out.layout, out.record!, ctx());
      expect(back.changed).toBe(true);
      expect(validateLayout(back.layout)).toEqual({ valid: true });
      expect(back.layout).not.toBeNull();
    }
    const tile = closeTile(abc(), 't2');
    const back = reopen(tile.layout, tile.record!, ctx());
    expect(back.layout).toEqual({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['a'], showing: 'a', size: 0.5 },
        { tile: 't2', tabs: ['b', 'e', 'f'], showing: 'e', size: 0.3, from: 't1' },
        { tile: 't3', tabs: ['c'], showing: 'c', size: 0.2 },
      ],
    });
  });

  // Owner ruling q-0004: a reopened tile comes back at its old size.
  it('a closed tile reopens at the share it had, from the neighbour only', () => {
    const out = closeTile(abc(), 't1');
    expect(out.record).toMatchObject({ kind: 'tile', neighbour: 't2', side: 'left', share: 0.5 });
    const back = reopen(out.layout, out.record!, ctx());
    expect(back.layout).toEqual({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['a'], showing: 'a', size: 0.5, from: 't2' },
        { tile: 't2', tabs: ['b', 'e', 'f'], showing: 'e', size: 0.3 },
        { tile: 't3', tabs: ['c'], showing: 'c', size: 0.2 },
      ],
    });
  });

  // A tile whose space went to a sibling tile comes back exactly; one whose space
  // went to a cross split (t1, t5 here) reopens as an edge of one of its tiles.
  it('closing then reopening a tile beside a sibling tile restores every size', () => {
    const nested: Layout = {
      split: 'row',
      items: [
        { tile: 't1', tabs: ['a'], showing: 'a', size: 0.25 },
        {
          split: 'column',
          size: 0.45,
          items: [
            { tile: 't2', tabs: ['b'], showing: 'b', size: 0.7 },
            { tile: 't3', tabs: ['c'], showing: 'c', size: 0.2 },
            { tile: 't4', tabs: ['d'], showing: 'd', size: 0.1 },
          ],
        },
        { tile: 't5', tabs: ['e'], showing: 'e', size: 0.3 },
      ],
    };
    const strip = (node: Layout): Layout =>
      'tile' in node
        ? { tile: node.tile, tabs: node.tabs, showing: node.showing, size: node.size }
        : { split: node.split, items: node.items.map(strip), size: node.size };
    for (const id of ['t2', 't3', 't4']) {
      const out = closeTile(nested, id);
      const back = reopen(out.layout, out.record!, ctx({ focusedTile: null }));
      expect(diffLayout(strip(back.layout!), strip(nested)), id).toEqual([]);
    }
  });

  it('a record without a share (stored before shares) halves the neighbour', () => {
    const legacy: TileRecord = { ...(closeTile(abc(), 't1').record as TileRecord) };
    delete legacy.share;
    const back = reopen(closeTile(abc(), 't1').layout, legacy, ctx());
    expect(back.layout).toEqual({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['a'], showing: 'a', size: 0.4, from: 't2' },
        { tile: 't2', tabs: ['b', 'e', 'f'], showing: 'e', size: 0.4 },
        { tile: 't3', tabs: ['c'], showing: 'c', size: 0.2 },
      ],
    });
  });

  it('the only tile records no share', () => {
    const out = closeTile({ tile: 't1', tabs: ['a'], showing: 'a' }, 't1');
    expect(out.record).not.toHaveProperty('share');
  });

  it('a tile reopens under a minted id when its old id is taken', () => {
    const one: Layout = { tile: 't2', tabs: ['a'], showing: 'a' };
    const out = reopen(
      one,
      { kind: 'tile', tile: 't2', tabs: ['b'], showing: 'b', neighbour: 't2', side: 'right' },
      ctx({ focusedTile: 't2' })
    );
    expect(out.layout).toEqual({
      split: 'row',
      items: [
        { tile: 't2', tabs: ['a'], showing: 'a', size: 0.5 },
        { tile: 't3', tabs: ['b'], showing: 'b', size: 0.5, from: 't2' },
      ],
    });
  });

  it('an infeasible edge split falls back to rule 7 with the recorded showing tab', () => {
    const tiny = { px: { width: 70, height: 600 }, minPx: 40 };
    const one: Layout = { tile: 't1', tabs: ['a'], showing: 'a' };
    const out = reopen(
      one,
      { kind: 'tile', tile: 't2', tabs: ['b', 'd'], showing: 'b', neighbour: 't1', side: 'right' },
      { focusedTile: 't1', dock: tiny }
    );
    expect(out.layout).toEqual({ tile: 't1', tabs: ['a', 'b', 'd'], showing: 'b' });
  });

  it('a tab record whose key is already open just shows it', () => {
    const out = reopen(
      abc(),
      { kind: 'tab', key: 'f', tile: 't2', index: 0, showing: false },
      ctx()
    );
    expect(out.changed).toBe(true);
    expect(out.layout).toEqual({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['a'], showing: 'a', size: 0.5 },
        { tile: 't2', tabs: ['b', 'e', 'f'], showing: 'f', size: 0.3 },
        { tile: 't3', tabs: ['c'], showing: 'c', size: 0.2 },
      ],
    });
  });

  it('reopen never mutates its input', () => {
    const before = abc();
    const copy = structuredClone(before);
    reopen(before, { kind: 'tab', key: 'x', tile: 't2', index: 1, showing: false }, ctx());
    reopen(
      before,
      { kind: 'tabs', records: [{ kind: 'tab', key: 'x', tile: 't9', index: 0, showing: true }] },
      ctx()
    );
    expect(before).toEqual(copy);
  });
});
