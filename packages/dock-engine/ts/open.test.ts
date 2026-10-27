// C-DROP: openItem (rule 7) and openSet (rules 15/16) replay every open and
// open-set case, leave valid layouts and never touch their input.
import { describe, expect, it } from 'vitest';
import { loadCases, runOpCase } from './cases.js';
import type { OpCase } from './cases.js';
import { validateLayout } from './layout.js';
import type { Layout } from './layout.js';
import { openItem, openSet } from './open.js';
import type { OpenContext } from './open.js';

const engine = { openItem, openSet };

describe('open cases', () => {
  const loaded = loadCases('open');
  it('has cases', () => expect(loaded.length).toBeGreaterThan(0));
  it.each(loaded.map((c) => [c.file, c]))('%s', (_file, c) => {
    expect(runOpCase(c.data as OpCase, engine)).toEqual([]);
  });
});

describe('open-set cases', () => {
  const loaded = loadCases('open-set');
  it('has cases', () => expect(loaded.length).toBeGreaterThan(0));
  it.each(loaded.map((c) => [c.file, c]))('%s', (_file, c) => {
    expect(runOpCase(c.data as OpCase, engine)).toEqual([]);
  });
});

const abc = (): Layout => ({
  split: 'row',
  items: [
    { tile: 't1', tabs: ['a'], showing: 'a', size: 0.5 },
    { tile: 't2', tabs: ['b', 'e'], showing: 'b', size: 0.3 },
    { tile: 't3', tabs: ['c'], showing: 'c', size: 0.2 },
  ],
});

const dock = { px: { width: 1000, height: 600 }, minPx: 40 };
const ctx = (extra: Partial<OpenContext> = {}): OpenContext => ({
  focusedTile: 't2',
  dock,
  ...extra,
});

describe('openItem', () => {
  it('leaves its input untouched and a valid layout', () => {
    const before = abc();
    const copy = structuredClone(before);
    const out = openItem(before, 'x', ctx());
    expect(before).toEqual(copy);
    expect(validateLayout(out.layout)).toEqual({ valid: true });
    expect(out.focus).toBe('t2');
  });

  it('an open key already showing changes nothing and focuses its tile', () => {
    const out = openItem(abc(), 'c', ctx());
    expect(out.changed).toBe(false);
    expect(out.layout).toEqual(abc());
    expect(out.focus).toBe('t3');
  });

  it('an infeasible picked target falls back to rule 7', () => {
    const tiny = { px: { width: 70, height: 600 }, minPx: 40 };
    const one: Layout = { tile: 't1', tabs: ['a'], showing: 'a' };
    const out = openItem(one, 'x', {
      focusedTile: 't1',
      dock: tiny,
      pickTarget: () => ({ tile: 't1', edge: 'right' }),
    });
    expect(out.layout).toEqual({ tile: 't1', tabs: ['a', 'x'], showing: 'x' });
    expect(out.focus).toBe('t1');
  });

  it('a missing focused tile opens into the first tile', () => {
    const out = openItem(abc(), 'x', ctx({ focusedTile: null }));
    expect(out.focus).toBe('t1');
    expect(out.changed).toBe(true);
  });
});

describe('openSet', () => {
  it('tabs on an empty dock make one tile', () => {
    const out = openSet(null, ['d', 'e'], 'tabs', undefined, ctx({ focusedTile: null }));
    expect(out.layout).toEqual({ tile: 't1', tabs: ['d', 'e'], showing: 'd' });
    expect(out.changed).toBe(true);
  });

  it('never duplicates a member that is already open', () => {
    const out = openSet(abc(), ['a', 'd'], 'tabs', undefined, ctx());
    expect(validateLayout(out.layout)).toEqual({ valid: true });
    expect(out.layout).not.toBeNull();
  });

  it('a side set in a column flattens into a valid layout', () => {
    const col: Layout = {
      split: 'column',
      items: [
        { tile: 't1', tabs: ['a'], showing: 'a', size: 0.5 },
        { tile: 't2', tabs: ['b'], showing: 'b', size: 0.5 },
      ],
    };
    const copy = structuredClone(col);
    const out = openSet(col, ['d', 'e', 'f'], 'side', undefined, ctx());
    expect(col).toEqual(copy);
    expect(out.changed).toBe(true);
    expect(validateLayout(out.layout)).toEqual({ valid: true });
  });

  it('layout skips members and records the dock it replaced', () => {
    const before = abc();
    const out = openSet(
      before,
      ['d', 'e'],
      'layout',
      { split: 'row', items: [{ tabs: ['d'] }, { tabs: ['e'] }] },
      { ...ctx(), skip: ['d', 'e'] }
    );
    expect(out.changed).toBe(false);
    expect(out.layout).toEqual(before);
    expect(out.skipped).toEqual(['d', 'e']);
  });
});
