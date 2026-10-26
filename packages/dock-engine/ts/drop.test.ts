// C-DROP: planDrop and replaceKey replay every drop and replace-key case, and
// the shared tree primitives behave as the other operations rely on.
import { describe, expect, it } from 'vitest';
import { loadCases, runOpCase } from './cases.js';
import type { OpCase } from './cases.js';
import { planDrop, removeTab, removeTile, replaceKey, splitTile, tileOf } from './drop.js';
import { validateLayout } from './layout.js';
import type { Layout } from './layout.js';

const engine = { planDrop, replaceKey };

describe('drop cases', () => {
  const loaded = loadCases('drop');
  it('has cases', () => expect(loaded.length).toBeGreaterThan(0));
  it.each(loaded.map((c) => [c.file, c]))('%s', (_file, c) => {
    expect(runOpCase(c.data as OpCase, engine)).toEqual([]);
  });

  it.each(loaded.map((c) => [c.file, c]))(
    '%s leaves a valid layout and its input untouched',
    (_file, c) => {
      const data = c.data as OpCase;
      if (data.op.fn !== 'planDrop') return;
      const before = JSON.stringify(data.before);
      const result = planDrop(data.before, data.op.source, data.op.target, data.op.dock);
      expect(JSON.stringify(data.before)).toBe(before);
      if (result.layout !== null) expect(validateLayout(result.layout)).toEqual({ valid: true });
    }
  );
});

describe('replace-key cases', () => {
  const loaded = loadCases('replace-key');
  it('has cases', () => expect(loaded.length).toBeGreaterThan(0));
  it.each(loaded.map((c) => [c.file, c]))('%s', (_file, c) => {
    expect(runOpCase(c.data as OpCase, engine)).toEqual([]);
  });
});

const abc = (): Layout => ({
  split: 'row',
  items: [
    { tile: 't1', tabs: ['a'], showing: 'a', size: 0.5 },
    { tile: 't2', tabs: ['b', 'e'], showing: 'e', size: 0.3, from: 't3' },
    { tile: 't3', tabs: ['c'], showing: 'c', size: 0.2 },
  ],
});

const dock = { px: { width: 1000, height: 600 }, minPx: 40 };

describe('primitives', () => {
  it('removeTile gives the space to the from subtree and names it', () => {
    const out = removeTile(abc(), 't2');
    expect(out.neighbour).toBe('t3');
    expect(out.side).toBe('left');
    expect(out.layout).toEqual({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['a'], showing: 'a', size: 0.5 },
        { tile: 't3', tabs: ['c'], showing: 'c', size: 0.5 },
      ],
    });
  });

  it('removeTile of the first tile gives the space to the next', () => {
    const out = removeTile(abc(), 't1');
    expect(out.neighbour).toBe('t2');
    expect(out.side).toBe('left');
  });

  it('removeTile clears a from that named the removed tile', () => {
    const out = removeTile(abc(), 't3');
    expect(out.neighbour).toBe('t2');
    expect(out.side).toBe('right');
    expect(tileOf(out.layout, 'b')).toEqual({
      tile: 't2',
      tabs: ['b', 'e'],
      showing: 'e',
      size: 0.5,
    });
  });

  it('removeTile of the only tile empties the dock', () => {
    const out = removeTile({ tile: 't1', tabs: ['a'], showing: 'a' }, 't1');
    expect(out).toEqual({ layout: null, neighbour: null, side: null });
  });

  it('removeTab of the showing tab shows the next, else the previous', () => {
    const out = removeTab(abc(), 'e');
    expect(tileOf(out.layout, 'b')?.showing).toBe('b');
  });

  it('splitTile refuses an infeasible split with null', () => {
    const tiny = { px: { width: 70, height: 600 }, minPx: 40 };
    const one: Layout = { tile: 't1', tabs: ['a'], showing: 'a' };
    const neu = { tile: 't2', tabs: ['b'], showing: 'b' };
    expect(splitTile(one, 't1', 'right', neu, tiny)).toBeNull();
    expect(splitTile(one, 't1', 'bottom', neu, tiny)).not.toBeNull();
  });

  it('an unknown target or source changes nothing', () => {
    expect(planDrop(abc(), { tab: 'zz' }, { tile: 't1', zone: 'centre' }, dock).changed).toBe(
      false
    );
    expect(planDrop(abc(), { new: 'x' }, { tile: 't9', zone: 'centre' }, dock).changed).toBe(false);
    expect(planDrop(abc(), { tile: 't9' }, { outer: 'left' }, dock).changed).toBe(false);
  });

  it('a border at the dock edge has no split and changes nothing', () => {
    expect(
      planDrop(abc(), { new: 'x' }, { border: { tile: 't1', side: 'left' } }, dock).changed
    ).toBe(false);
  });

  it('a new item on an empty dock becomes the only tile', () => {
    expect(planDrop(null, { new: 'x' }, { outer: 'left' }, dock)).toEqual({
      layout: { tile: 't1', tabs: ['x'], showing: 'x' },
      changed: true,
    });
  });

  it('a tab dropped back where it is changes nothing', () => {
    const before: Layout = { tile: 't1', tabs: ['a', 'b'], showing: 'b' };
    const out = planDrop(before, { tab: 'b' }, { tile: 't1', zone: 'tab', index: 2 }, dock);
    expect(out).toEqual({ layout: before, changed: false });
  });
});
