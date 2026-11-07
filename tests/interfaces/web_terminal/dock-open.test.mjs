// @ts-check
// @vitest-environment node
/**
 * The dock's placement rules (dock-open.js), as pure operations on a layout:
 *   npx vitest run tests/interfaces/web_terminal/dock-open.test.mjs
 *
 * Every tile holds one panel. The open rule: a panel takes the service tile
 * the operator used last, replacing the panel that tile showed, and nothing
 * resizes; a panel already docked is focused where it is; open-at splits
 * where a drop said and moves an already-docked panel; with no service tile
 * the first one opens left of the terminal at 0.6 / 0.4; a set opens side by
 * side; an arrangement giving a tile several panels is refused by the
 * policy's name; the terminal reopens at the right edge at 0.4 and closes as
 * its tile.
 */

import { describe, expect, test } from 'vitest';

import {
  SERVICE_SHARE,
  TERMINAL_SHARE,
  arrange,
  closeKey,
  closeKeys,
  defaultLayout,
  firstTile,
  focusKey,
  openAt,
  openPanel,
  openPanelSet,
  reopenTerminal,
  serviceTileTarget,
} from '../../../src/osprey/interfaces/web_terminal/static/js/dock-open.js';
import {
  ONE_PANEL_PER_TILE,
  crowdedTile,
} from '../../../src/osprey/interfaces/web_terminal/static/js/dock-policy.js';
import {
  MIN_PX,
  TOO_MANY_FOR_SIDE,
  keysOf,
  tileOf,
  validateLayout,
} from '/static/vendor/dock-engine.mjs';

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */

const DOCK = { px: { width: 1000, height: 600 }, minPx: MIN_PX };

/** @param {string} id @param {string[]} tabs @param {string} [showing] @returns {any} */
function tile(id, tabs, showing = tabs[0]) {
  return { tile: id, tabs, showing };
}

/** @param {any[]} items @returns {any} */
function row(...items) {
  const size = 1 / items.length;
  return { split: 'row', items: items.map((item) => ({ ...item, size })) };
}

const TERMINAL = tile('tt', ['terminal']);

/**
 * @param {Partial<{ focusedTile: string|null, lastServiceTile: string|null }>} [focus]
 */
function ctx(focus = {}) {
  return { focusedTile: null, lastServiceTile: null, dock: DOCK, ...focus };
}

/** Every tile of a layout, in tree order. @param {any} node @returns {any[]} */
function tiles(node) {
  if (node === null) return [];
  return 'tabs' in node ? [node] : node.items.flatMap(tiles);
}

describe('serviceTileTarget', () => {
  const layout = row(tile('a', ['alpha']), tile('b', ['beta']), TERMINAL);

  test('the focused tile when it shows a service', () => {
    expect(serviceTileTarget(layout, ctx({ focusedTile: 'b' }))?.tile).toBe('b');
  });

  test('the last service tile when the terminal is focused', () => {
    expect(serviceTileTarget(layout, ctx({ focusedTile: 'tt', lastServiceTile: 'b' }))?.tile).toBe(
      'b'
    );
  });

  test('the first service tile in tree order when neither is known', () => {
    expect(serviceTileTarget(layout, ctx())?.tile).toBe('a');
  });

  test('null when no tile shows a service', () => {
    expect(serviceTileTarget(TERMINAL, ctx({ focusedTile: 'tt' }))).toBeNull();
    expect(serviceTileTarget(null, ctx())).toBeNull();
  });
});

describe('firstTile — the classic 60/40 split', () => {
  test('opens left of the terminal at the service share', () => {
    const out = firstTile(TERMINAL, 'alpha', DOCK);
    expect(out.changed).toBe(true);
    const [left, right] = tiles(out.layout);
    expect(left.tabs).toEqual(['alpha']);
    expect(right.tabs).toEqual(['terminal']);
    expect(left.size).toBeCloseTo(SERVICE_SHARE, 3);
    expect(right.size).toBeCloseTo(1 - SERVICE_SHARE, 3);
    expect(out.focus).toBe(left.tile);
    expect(validateLayout(out.layout).valid).toBe(true);
  });

  test('a terminal already sharing a row keeps its siblings proportional', () => {
    const layout = row(tile('x', ['xi']), TERMINAL);
    const out = /** @type {any} */ (firstTile(layout, 'alpha', DOCK).layout);
    expect(tiles(out).map((t) => t.tabs[0])).toEqual(['xi', 'alpha', 'terminal']);
    const alpha = tiles(out)[1];
    expect(alpha.size).toBeCloseTo(SERVICE_SHARE, 3);
  });

  test('opens against the left edge when the terminal tile is closed too', () => {
    const out = firstTile(tile('x', ['xi']), 'alpha', DOCK);
    expect(tiles(out.layout).map((t) => t.tabs[0])).toEqual(['alpha', 'xi']);
  });

  test('is the lone tile of an empty dock', () => {
    const out = firstTile(null, 'alpha', DOCK);
    expect(out).toEqual({
      layout: { tile: 't1', tabs: ['alpha'], showing: 'alpha' },
      changed: true,
      focus: 't1',
    });
  });
});

describe('openPanel — the open rule replaces the panel of the service tile used last', () => {
  test('the focused service tile shows the panel instead; no tile resizes and no other changes', () => {
    const layout = row(tile('a', ['alpha']), tile('b', ['beta']), TERMINAL);
    const before = structuredClone(layout);
    const out = openPanel(layout, 'gamma', ctx({ focusedTile: 'b' }));
    expect(out.changed).toBe(true);
    expect(out.focus).toBe('b');
    // Tile b now shows gamma alone; beta left the dock (its sidebar row
    // opens it again), and the other tiles are as they were.
    expect(tiles(out.layout)).toEqual([
      { tile: 'a', tabs: ['alpha'], showing: 'alpha', size: 1 / 3 },
      { tile: 'b', tabs: ['gamma'], showing: 'gamma', size: 1 / 3 },
      { tile: 'tt', tabs: ['terminal'], showing: 'terminal', size: 1 / 3 },
    ]);
    expect(keysOf(out.layout)).toEqual(['alpha', 'gamma', 'terminal']);
    expect(crowdedTile(out.layout)).toBeNull();
    expect(validateLayout(out.layout).valid).toBe(true);
    expect(layout).toEqual(before);
  });

  test('a panel already docked changes nothing; focusKey focuses its tile where it is', () => {
    const layout = row(tile('a', ['alpha']), tile('b', ['beta']), TERMINAL);
    expect(openPanel(layout, 'alpha', ctx({ focusedTile: 'b' }))).toEqual({
      layout,
      changed: false,
    });
    const focused = focusKey(layout, 'alpha');
    expect(focused).toEqual({ layout, changed: false, focus: 'a' });
    expect(focused.layout).toBe(layout);
    // A key the dock does not hold has no tile to focus.
    expect(focusKey(layout, 'gamma')).toEqual({ layout, changed: false });
  });

  test('with the terminal focused, the last service tile takes the panel — never the terminal', () => {
    const layout = row(tile('a', ['alpha']), tile('b', ['beta']), TERMINAL);
    const out = openPanel(layout, 'okf', ctx({ focusedTile: 'tt', lastServiceTile: 'a' }));
    expect(tileOf(out.layout, 'terminal')).toEqual({
      tile: 'tt',
      tabs: ['terminal'],
      showing: 'terminal',
      size: 1 / 3,
    });
    expect(tileOf(out.layout, 'okf')).toMatchObject({ tile: 'a', tabs: ['okf'], showing: 'okf' });
    expect(tileOf(out.layout, 'beta')?.tile).toBe('b');
    expect(out.focus).toBe('a');
  });

  test('with no service tile it opens the first tile left of the terminal at the classic split', () => {
    const out = openPanel(TERMINAL, 'alpha', ctx({ focusedTile: 'tt' }));
    const [left, right] = tiles(out.layout);
    expect([left.tabs, right.tabs]).toEqual([['alpha'], ['terminal']]);
    expect(left.size).toBeCloseTo(SERVICE_SHARE, 3);
    expect(right.size).toBeCloseTo(1 - SERVICE_SHARE, 3);
    expect(out.focus).toBe(left.tile);
  });

  test('an empty dock gets one tile', () => {
    expect(openPanel(null, 'alpha', ctx())).toEqual({
      layout: { tile: 't1', tabs: ['alpha'], showing: 'alpha' },
      changed: true,
      focus: 't1',
    });
  });
});

describe('openAt — placement at an explicit position', () => {
  const layout = row(tile('a', ['alpha']), TERMINAL);

  test('splits the reference tile on the given side', () => {
    const out = openAt(layout, 'beta', { referenceTile: 'a', direction: 'bottom' }, ctx());
    expect(out.changed).toBe(true);
    const column = /** @type {any} */ (out.layout).items[0];
    expect(column.split).toBe('column');
    expect(column.items.map((/** @type {any} */ t) => t.tabs[0])).toEqual(['alpha', 'beta']);
  });

  test('splits against the dock edge when no tile is named', () => {
    const out = openAt(layout, 'beta', { direction: 'right' }, ctx());
    expect(tiles(out.layout).map((t) => t.tabs[0])).toEqual(['alpha', 'terminal', 'beta']);
  });

  test('moves a docked panel to the position, never duplicating it', () => {
    const out = openAt(layout, 'alpha', { referenceTile: 'tt', direction: 'right' }, ctx());
    expect(out.changed).toBe(true);
    expect(tiles(out.layout).map((t) => t.tabs[0])).toEqual(['terminal', 'alpha']);
    expect(keysOf(out.layout).filter((k) => k === 'alpha')).toHaveLength(1);
  });

  test('a position naming the panel’s own tile is a no-op', () => {
    expect(openAt(layout, 'alpha', { referenceTile: 'a', direction: 'right' }, ctx()).changed).toBe(
      false
    );
  });

  test('a docked panel with no position is a no-op', () => {
    expect(openAt(layout, 'alpha', null, ctx({ focusedTile: 'tt' })).changed).toBe(false);
  });

  test('no position: beside the focused tile, else the first tile', () => {
    const beside = openAt(layout, 'beta', null, ctx({ focusedTile: 'tt' }));
    expect(tiles(beside.layout).map((t) => t.tabs[0])).toEqual(['alpha', 'terminal', 'beta']);
    const first = openAt(TERMINAL, 'beta', null, ctx());
    expect(tiles(first.layout).map((t) => t.tabs[0])).toEqual(['beta', 'terminal']);
    expect(openAt(null, 'beta', null, ctx()).layout).toEqual(tile('t1', ['beta']));
  });

  test('an unknown reference tile or side changes nothing', () => {
    expect(openAt(layout, 'beta', { referenceTile: 'zz', direction: 'right' }, ctx()).changed).toBe(
      false
    );
    expect(
      openAt(
        layout,
        'beta',
        { referenceTile: 'a', direction: /** @type {any} */ ('within') },
        ctx()
      ).changed
    ).toBe(false);
  });
});

describe('openPanelSet — side by side, each panel in a tile of its own', () => {
  test('the set shares the span of the service tile used last; the terminal keeps its share', () => {
    const layout = row(tile('a', ['alpha']), TERMINAL);
    const out = openPanelSet(
      layout,
      ['beta', 'gamma'],
      'side',
      undefined,
      ctx({ focusedTile: 'a' })
    );
    expect(out.changed).toBe(true);
    const all = tiles(out.layout);
    expect(all.map((t) => t.tabs)).toEqual([['alpha'], ['beta'], ['gamma'], ['terminal']]);
    expect(all[3]).toMatchObject({ tile: 'tt', size: 0.5 });
    expect(all.slice(0, 3).reduce((sum, t) => sum + t.size, 0)).toBeCloseTo(0.5, 3);
    expect(out.focus).toBe(tileOf(out.layout, 'beta')?.tile);
    expect(crowdedTile(out.layout)).toBeNull();
    expect(validateLayout(out.layout).valid).toBe(true);
  });

  test('with no service tile the first member opens beside the terminal at 60/40, the rest beside it', () => {
    const out = openPanelSet(TERMINAL, ['alpha', 'beta'], 'side', undefined, ctx());
    expect(out.changed).toBe(true);
    const all = tiles(out.layout);
    expect(all.map((t) => t.tabs)).toEqual([['alpha'], ['beta'], ['terminal']]);
    expect(all[2].size).toBeCloseTo(1 - SERVICE_SHARE, 3);
    expect(all[0].size + all[1].size).toBeCloseTo(SERVICE_SHARE, 3);
    expect(out.focus).toBe(all[0].tile);
    expect(crowdedTile(out.layout)).toBeNull();
  });

  test('a first tile too narrow to share is refused by the engine’s name; nothing changes', () => {
    // At 120 x 60 the first tile (72 px wide) cannot hold two tiles of 40 px.
    const narrow = { ...ctx(), dock: { px: { width: 120, height: 60 }, minPx: MIN_PX } };
    const out = openPanelSet(TERMINAL, ['alpha', 'beta'], 'side', undefined, narrow);
    expect(out).toEqual({ layout: TERMINAL, changed: false, refusal: TOO_MANY_FOR_SIDE });
    expect(out.layout).toBe(TERMINAL);
  });
});

describe('arrange — a given layout replaces the service tiles', () => {
  const layout = row(tile('a', ['alpha']), TERMINAL);

  test.each([
    ['a split', { split: 'row', items: [{ tabs: ['ariel', 'lattice'] }, { tabs: ['okf'] }] }],
    ['a root tile', { tabs: ['ariel', 'lattice'], showing: 'lattice' }],
  ])('%s giving a tile several panels is refused by the policy’s name', (_name, given) => {
    const set = /** @type {any} */ (given);
    const refused = { layout, changed: false, refusal: ONE_PANEL_PER_TILE };
    expect(arrange(layout, set, ctx({ focusedTile: 'a' }))).toEqual(refused);
    // An open_set "as layout" is the same arrangement.
    expect(openPanelSet(layout, ['ariel', 'lattice'], 'layout', set, ctx())).toEqual(refused);
  });

  test('a layout of one panel per tile replaces the services; the terminal keeps its tile', () => {
    const given = { split: 'column', items: [{ tabs: ['ariel'] }, { tabs: ['lattice'] }] };
    const out = arrange(layout, /** @type {any} */ (given), ctx({ focusedTile: 'a' }));
    expect(out.changed).toBe(true);
    expect(out.refusal).toBeUndefined();
    expect(tiles(out.layout).map((t) => t.tabs)).toEqual([['ariel'], ['lattice'], ['terminal']]);
    expect(tileOf(out.layout, 'terminal')?.tile).toBe('tt');
    expect(out.pruned).toEqual(['alpha']);
    expect(crowdedTile(out.layout)).toBeNull();
  });
});

describe('the terminal', () => {
  test('reopens at the dock’s right edge at its share', () => {
    const layout = row(tile('a', ['alpha']), tile('b', ['beta']));
    const out = reopenTerminal(layout, DOCK);
    expect(out.changed).toBe(true);
    const all = tiles(out.layout);
    expect(all.map((t) => t.tabs[0])).toEqual(['alpha', 'beta', 'terminal']);
    expect(all[2].size).toBeCloseTo(TERMINAL_SHARE, 3);
    expect(out.focus).toBe(all[2].tile);
  });

  test('reopens as the lone tile of an empty dock', () => {
    expect(reopenTerminal(null, DOCK).layout).toEqual(tile('t1', ['terminal']));
  });

  test('an open terminal changes nothing', () => {
    const layout = row(tile('a', ['alpha']), TERMINAL);
    expect(reopenTerminal(layout, DOCK)).toEqual({ layout, changed: false, focus: 'tt' });
  });

  test('closes as its tile, handing its space to its neighbour', () => {
    const layout = row(tile('a', ['alpha']), TERMINAL);
    const out = closeKey(layout, 'terminal');
    expect(out.changed).toBe(true);
    expect(out.layout).toEqual(tile('a', ['alpha']));
    expect(closeKey(out.layout, 'terminal').changed).toBe(false);
  });
});

describe('closeKeys and the default', () => {
  test('closes several keys as one change', () => {
    const layout = row(tile('a', ['alpha']), tile('b', ['beta']), TERMINAL);
    const out = closeKeys(layout, ['alpha', 'ghost', 'beta']);
    expect(out.changed).toBe(true);
    expect(out.layout).toEqual(TERMINAL);
    expect(closeKeys(layout, ['ghost'])).toEqual({ layout, changed: false });
  });

  test('the default is the terminal alone', () => {
    expect(defaultLayout()).toEqual({ tile: 't1', tabs: ['terminal'], showing: 'terminal' });
  });
});
