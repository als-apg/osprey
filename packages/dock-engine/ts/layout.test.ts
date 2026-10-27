// C-LAYOUT: the layout tree shared by both docks, both agents, collections,
// presets and persistence. These tests pin the validator's named reasons, the
// normalised form and the tile-id minting rule.
import { describe, expect, it } from 'vitest';
import {
  LAYOUT_REASONS,
  isTile,
  keysOf,
  mintTileId,
  normalize,
  normalizeCollection,
  tileIdsOf,
  validateLayout,
} from './layout.js';
import type { CollectionLayout, Layout, Split, Tile } from './layout.js';

function tile(id: string, tabs: string[], extra: Partial<Tile> = {}): Tile {
  return { tile: id, tabs, showing: tabs[0] ?? '', ...extra };
}

function split(dir: 'row' | 'column', items: Layout[], size?: number): Split {
  return size === undefined ? { split: dir, items } : { split: dir, items, size };
}

function reasonOf(input: unknown, options?: Parameters<typeof validateLayout>[1]): string {
  const result = validateLayout(input, options);
  return result.valid ? 'valid' : result.reason;
}

function sizes(layout: Layout): number[] {
  return isTile(layout) ? [] : layout.items.map((item) => item.size ?? Number.NaN);
}

describe('validateLayout (dock form)', () => {
  it('accepts a root tile', () => {
    expect(validateLayout(tile('t1', ['a']))).toEqual({ valid: true });
  });

  it('accepts a split with partial sizes that leave a remainder', () => {
    const layout = split('row', [
      tile('t1', ['a'], { size: 0.5 }),
      tile('t2', ['b']),
      tile('t3', ['c']),
    ]);
    expect(reasonOf(layout)).toBe('valid');
  });

  it('accepts a from that names another tile', () => {
    const layout = split('row', [tile('t1', ['a']), tile('t2', ['b'], { from: 't1' })]);
    expect(reasonOf(layout)).toBe('valid');
  });

  it('refuses a key in two tiles as duplicate_key', () => {
    const layout = split('row', [tile('t1', ['a']), tile('t2', ['b', 'a'])]);
    const result = validateLayout(layout);
    expect(result).toEqual({
      valid: false,
      reason: LAYOUT_REASONS.duplicateKey,
      where: 'layout.items[1].tabs[1]',
    });
  });

  it('refuses a key twice in one tile as duplicate_key', () => {
    expect(reasonOf(tile('t1', ['a', 'a']))).toBe('duplicate_key');
  });

  it('refuses a showing that is not one of the tabs', () => {
    expect(reasonOf({ tile: 't1', tabs: ['a'], showing: 'b' })).toBe('showing_not_in_tabs');
  });

  it('refuses a dock tile without showing', () => {
    expect(reasonOf({ tile: 't1', tabs: ['a'] })).toBe('showing_not_in_tabs');
  });

  it('refuses empty tabs', () => {
    expect(reasonOf({ tile: 't1', tabs: [], showing: 'a' })).toBe('empty_tabs');
  });

  it('refuses a split with one item or none', () => {
    expect(reasonOf(split('row', [tile('t1', ['a'])]))).toBe('split_too_few_items');
    expect(reasonOf({ split: 'column', items: [] })).toBe('split_too_few_items');
  });

  it.each([
    ['zero', 0],
    ['negative', -0.2],
    ['above one', 1.5],
    ['NaN', Number.NaN],
    ['a string', '0.5'],
  ])('refuses a %s size as bad_size', (_label, size) => {
    const layout = { split: 'row', items: [{ ...tile('t1', ['a']), size }, tile('t2', ['b'])] };
    expect(reasonOf(layout)).toBe('bad_size');
  });

  it('refuses sibling fractions that do not sum to 1', () => {
    const layout = split('row', [
      tile('t1', ['a'], { size: 0.5 }),
      tile('t2', ['b'], { size: 0.4 }),
    ]);
    expect(validateLayout(layout)).toEqual({
      valid: false,
      reason: 'bad_size',
      where: 'layout.items',
    });
  });

  it('accepts sibling fractions within 0.001 of 1', () => {
    const layout = split('row', [
      tile('t1', ['a'], { size: 0.3335 }),
      tile('t2', ['b'], { size: 0.667 }),
    ]);
    expect(reasonOf(layout)).toBe('valid');
  });

  it('refuses given sizes that leave nothing for the missing ones', () => {
    const layout = split('row', [
      tile('t1', ['a'], { size: 0.6 }),
      tile('t2', ['b'], { size: 0.4 }),
      tile('t3', ['c']),
    ]);
    expect(reasonOf(layout)).toBe('bad_size');
  });

  it('refuses a from that names no tile, or the tile itself', () => {
    expect(reasonOf(split('row', [tile('t1', ['a']), tile('t2', ['b'], { from: 't9' })]))).toBe(
      'unknown_from'
    );
    expect(reasonOf(tile('t1', ['a'], { from: 't1' }))).toBe('unknown_from');
  });

  it('refuses the old slot-leaf collection shape', () => {
    const old = { split: 'row', items: [{ slot: 'a' }, { slot: 'b', size: 2 }], active: 'a' };
    expect(reasonOf(old)).toBe('old_slot_shape');
    expect(reasonOf(old, { form: 'collection' })).toBe('old_slot_shape');
  });

  it('refuses the old {key,size} leaf shape', () => {
    const old = {
      split: 'row',
      items: [{ key: 'a', size: 0.5 }, { items: [{ key: 'b' }, { key: 'c' }] }],
    };
    expect(reasonOf(old)).toBe('old_key_shape');
  });

  it('names a nested old leaf before any other violation', () => {
    const old = {
      split: 'row',
      items: [tile('t1', ['a']), { split: 'column', items: [{ slot: 'x' }] }],
    };
    expect(reasonOf(old)).toBe('old_slot_shape');
  });

  it('refuses a duplicate tile id', () => {
    expect(reasonOf(split('row', [tile('t1', ['a']), tile('t1', ['b'])]))).toBe('duplicate_tile');
  });

  it('refuses an unknown field and a bad split direction', () => {
    expect(reasonOf({ ...tile('t1', ['a']), extra: 1 })).toBe('unknown_field');
    expect(reasonOf({ split: 'diagonal', items: [tile('t1', ['a']), tile('t2', ['b'])] })).toBe(
      'bad_split'
    );
    expect(reasonOf({ items: [tile('t1', ['a']), tile('t2', ['b'])] })).toBe('bad_split');
  });

  it('refuses non-layouts', () => {
    expect(reasonOf(null)).toBe('not_layout');
    expect(reasonOf([])).toBe('not_layout');
    expect(reasonOf({})).toBe('not_layout');
    expect(reasonOf({ tabs: ['a'], showing: 'a' })).toBe('bad_tile_id');
    expect(reasonOf({ tile: 't1', tabs: [''], showing: '' })).toBe('bad_key');
  });

  it('has no depth limit', () => {
    let layout: Layout = tile('t0', ['k0']);
    for (let i = 1; i <= 40; i += 1) {
      layout = split(i % 2 === 0 ? 'row' : 'column', [layout, tile(`t${i}`, [`k${i}`])]);
    }
    expect(reasonOf(layout)).toBe('valid');
  });

  it('every reason it can return is a named constant', () => {
    const names = new Set(Object.values(LAYOUT_REASONS));
    for (const input of [
      null,
      tile('t1', ['a', 'a']),
      { tile: 't1', tabs: [], showing: 'a' },
      { split: 'row', items: [{ slot: 'a' }] },
    ]) {
      const result = validateLayout(input);
      expect(result.valid).toBe(false);
      if (!result.valid) expect(names.has(result.reason)).toBe(true);
    }
  });
});

describe('validateLayout (collection form)', () => {
  const opts = { form: 'collection' as const };

  it('accepts weights, a missing showing and a root tile', () => {
    const layout = {
      split: 'row',
      items: [
        { tabs: [0, 1], size: 2 },
        { tabs: [2], size: 3 },
      ],
    };
    expect(reasonOf(layout, opts)).toBe('valid');
    expect(reasonOf({ tabs: [0] }, opts)).toBe('valid');
    expect(reasonOf({ tabs: ['terminal', 'p1'], showing: 'p1' }, opts)).toBe('valid');
  });

  it('refuses tile ids and from in the collection form', () => {
    expect(reasonOf({ tile: 't1', tabs: [0] }, opts)).toBe('unknown_field');
    expect(reasonOf({ tabs: [0], from: 't1' }, opts)).toBe('unknown_field');
  });

  it('refuses a slot member field as the old shape', () => {
    expect(reasonOf({ tabs: [0], slot: 'a' }, opts)).toBe('old_slot_shape');
  });

  it('refuses a non-positive weight', () => {
    const layout = { split: 'row', items: [{ tabs: [0], size: 0 }, { tabs: [1] }] };
    expect(reasonOf(layout, opts)).toBe('bad_size');
  });

  it('refuses a showing outside tabs and a repeated member', () => {
    expect(reasonOf({ tabs: [0, 1], showing: 2 }, opts)).toBe('showing_not_in_tabs');
    expect(reasonOf({ split: 'row', items: [{ tabs: [0] }, { tabs: [0] }] }, opts)).toBe(
      'duplicate_key'
    );
  });

  it('refuses a bad member index', () => {
    expect(reasonOf({ tabs: [-1] }, opts)).toBe('bad_key');
    expect(reasonOf({ tabs: [1.5] }, opts)).toBe('bad_key');
  });

  it('with a member count, places every member exactly once', () => {
    const layout = { split: 'row', items: [{ tabs: [0] }, { tabs: [2] }] };
    expect(reasonOf(layout, { form: 'collection', members: 3 })).toBe('member_missing');
    expect(reasonOf(layout, { form: 'collection', members: 2 })).toBe('bad_key');
    const full = { split: 'row', items: [{ tabs: [0, 1] }, { tabs: [2] }] };
    expect(reasonOf(full, { form: 'collection', members: 3 })).toBe('valid');
  });
});

describe('normalize', () => {
  it('fills missing sizes with an equal share of the remainder', () => {
    const out = normalize(
      split('row', [tile('t1', ['a'], { size: 0.5 }), tile('t2', ['b']), tile('t3', ['c'])])
    );
    expect(sizes(out)).toEqual([0.5, 0.25, 0.25]);
  });

  it('fills an all-missing split with equal shares', () => {
    const out = normalize(
      split('column', [tile('t1', ['a']), tile('t2', ['b']), tile('t3', ['c'])])
    );
    for (const s of sizes(out)) expect(s).toBeCloseTo(1 / 3, 9);
  });

  it('renormalises sizes that are within tolerance', () => {
    const out = normalize(
      split('row', [tile('t1', ['a'], { size: 0.3335 }), tile('t2', ['b'], { size: 0.667 })])
    );
    const [a = 0, b = 0] = sizes(out);
    expect(a + b).toBeCloseTo(1, 12);
  });

  it('flattens a same-direction split, scaling its items by its share', () => {
    const out = normalize(
      split('row', [
        tile('t1', ['a'], { size: 0.5 }),
        split('row', [tile('t2', ['b'], { size: 0.4 }), tile('t3', ['c'], { size: 0.6 })], 0.5),
      ])
    );
    expect(isTile(out)).toBe(false);
    if (isTile(out)) return;
    expect(out.split).toBe('row');
    expect(out.items.map((i) => (isTile(i) ? i.tile : 'split'))).toEqual(['t1', 't2', 't3']);
    const [a = 0, b = 0, c = 0] = sizes(out);
    expect(a).toBeCloseTo(0.5, 9);
    expect(b).toBeCloseTo(0.2, 9);
    expect(c).toBeCloseTo(0.3, 9);
  });

  it('keeps a split of the other direction', () => {
    const out = normalize(
      split('row', [tile('t1', ['a']), split('column', [tile('t2', ['b']), tile('t3', ['c'])])])
    );
    expect(isTile(out)).toBe(false);
    if (isTile(out)) return;
    expect(out.items).toHaveLength(2);
    expect(out.items[1]).toMatchObject({ split: 'column', size: 0.5 });
  });

  it('collapses a one-item split; the item takes the split size', () => {
    const out = normalize(
      split('row', [tile('t1', ['a'], { size: 0.7 }), split('column', [tile('t2', ['b'])], 0.3)])
    );
    expect(isTile(out)).toBe(false);
    if (isTile(out)) return;
    expect(out.items[1]).toEqual({ tile: 't2', tabs: ['b'], showing: 'b', size: 0.3 });
  });

  it('collapses a one-item root split to a root tile without a size', () => {
    expect(normalize(split('row', [tile('t1', ['a'], { size: 1 })]))).toEqual(tile('t1', ['a']));
  });

  it('flattens through a collapsed child that exposes a same-direction split', () => {
    const inner = split('row', [tile('t2', ['b']), tile('t3', ['c'])]);
    const out = normalize(split('row', [tile('t1', ['a']), split('column', [inner])]));
    expect(isTile(out)).toBe(false);
    if (isTile(out)) return;
    expect(out.items.map((i) => (isTile(i) ? i.tile : 'split'))).toEqual(['t1', 't2', 't3']);
    const [a = 0, b = 0, c = 0] = sizes(out);
    expect(a).toBeCloseTo(0.5, 9);
    expect(b).toBeCloseTo(0.25, 9);
    expect(c).toBeCloseTo(0.25, 9);
  });

  it('clears a from whose tile is gone', () => {
    const out = normalize(split('row', [tile('t1', ['a']), tile('t2', ['b'], { from: 't9' })]));
    expect(isTile(out)).toBe(false);
    if (isTile(out)) return;
    expect(out.items[1]).not.toHaveProperty('from');
  });

  it('does not mutate its input and produces a valid layout', () => {
    const input = split('row', [
      tile('t1', ['a']),
      split('row', [tile('t2', ['b'], { from: 't1' }), tile('t3', ['c'])]),
    ]);
    const before = JSON.parse(JSON.stringify(input)) as unknown;
    const out = normalize(input);
    expect(input).toEqual(before);
    expect(validateLayout(out)).toEqual({ valid: true });
    expect(normalize(out)).toEqual(out);
  });

  it('handles deep nesting', () => {
    let layout: Layout = tile('t0', ['k0']);
    for (let i = 1; i <= 40; i += 1) {
      layout = split(i % 2 === 0 ? 'row' : 'column', [layout, tile(`t${i}`, [`k${i}`])]);
    }
    expect(validateLayout(normalize(layout))).toEqual({ valid: true });
  });
});

describe('normalizeCollection', () => {
  it('turns weights into fractions and defaults showing to tabs[0]', () => {
    const input: CollectionLayout = {
      split: 'column',
      items: [
        { tabs: [0, 1], size: 1 },
        { tabs: [2], size: 3 },
      ],
    };
    expect(normalizeCollection(input)).toEqual({
      split: 'column',
      items: [
        { tabs: [0, 1], showing: 0, size: 0.25 },
        { tabs: [2], showing: 2, size: 0.75 },
      ],
    });
  });

  it('counts a missing weight as 1', () => {
    const out = normalizeCollection({
      split: 'row',
      items: [{ tabs: [0], size: 2 }, { tabs: [1] }],
    });
    expect('items' in out ? out.items.map((i) => i.size) : []).toEqual([2 / 3, 1 / 3]);
  });

  it('keeps a root tile', () => {
    expect(normalizeCollection({ tabs: ['p1', 'terminal'], showing: 'terminal' })).toEqual({
      tabs: ['p1', 'terminal'],
      showing: 'terminal',
    });
  });
});

describe('mintTileId', () => {
  it('mints t1 for an empty layout', () => {
    expect(mintTileId(null)).toBe('t1');
  });

  it('mints one past the largest numeric suffix', () => {
    const layout = split('row', [
      tile('t2', ['a']),
      split('column', [tile('t10', ['b']), tile('x7', ['c']), tile('t03', ['d'])]),
    ]);
    expect(mintTileId(layout)).toBe('t11');
  });

  it('ignores ids that are not t<n>', () => {
    expect(mintTileId(split('row', [tile('main', ['a']), tile('t', ['b'])]))).toBe('t1');
  });

  it('reuses ids freed by a close', () => {
    expect(mintTileId(tile('t1', ['a']))).toBe('t2');
  });
});

describe('helpers', () => {
  it('lists tile ids and keys in tree order', () => {
    const layout = split('row', [
      tile('t1', ['a', 'b']),
      split('column', [tile('t2', ['c']), tile('t3', ['d'])]),
    ]);
    expect(tileIdsOf(layout)).toEqual(['t1', 't2', 't3']);
    expect(keysOf(layout)).toEqual(['a', 'b', 'c', 'd']);
  });
});
