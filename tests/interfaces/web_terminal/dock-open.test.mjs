// @ts-check
// @vitest-environment node
/**
 * The dock's placement rules (dock-open.js), as pure operations on a layout:
 *   npx vitest run tests/interfaces/web_terminal/dock-open.test.mjs
 *
 * One panel per tile: an activation shows the panel in place of the one the
 * target service tile shows; open-beside splits a new tile right of it;
 * open-at splits where a drop said and moves an already-docked panel; with
 * no service tile the first one opens left of the terminal at 0.6 / 0.4; the
 * terminal reopens at the right edge at 0.4 and closes as its tile.
 */

import { describe, expect, test } from 'vitest';

import {
  SERVICE_SHARE,
  TERMINAL_SHARE,
  closeKey,
  closeKeys,
  defaultLayout,
  firstTile,
  openAt,
  openBeside,
  reopenTerminal,
  serviceTileTarget,
  showIn,
} from '../../../src/osprey/interfaces/web_terminal/static/js/dock-open.js';
import { MIN_PX, keysOf, tileOf, validateLayout } from '/static/vendor/dock-engine.mjs';

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

describe('showIn — an activation takes over the target service tile', () => {
  test('replaces the panel the target tile shows; the evicted key leaves the layout', () => {
    const layout = row(tile('a', ['alpha']), TERMINAL);
    const out = showIn(layout, 'beta', ctx({ focusedTile: 'a' }));
    expect(out.changed).toBe(true);
    expect(keysOf(out.layout)).toEqual(['beta', 'terminal']);
    expect(tileOf(out.layout, 'beta')?.tile).toBe('a');
    expect(out.focus).toBe('a');
  });

  test('a panel already docked is shown where it is, nothing moves', () => {
    const layout = row(tile('a', ['alpha']), tile('b', ['beta']), TERMINAL);
    const out = showIn(layout, 'alpha', ctx({ focusedTile: 'b' }));
    expect(out).toEqual({ layout, changed: false, focus: 'a' });
  });

  test('with the terminal focused, the last service tile is taken over — never the terminal', () => {
    const layout = row(tile('a', ['alpha']), TERMINAL);
    const out = showIn(layout, 'okf', ctx({ focusedTile: 'tt', lastServiceTile: 'a' }));
    expect(tileOf(out.layout, 'terminal')?.tabs).toEqual(['terminal']);
    expect(tileOf(out.layout, 'okf')?.tile).toBe('a');
    expect(keysOf(out.layout)).not.toContain('alpha');
  });

  test('with no service tile it opens the first tile left of the terminal', () => {
    const out = showIn(TERMINAL, 'alpha', ctx({ focusedTile: 'tt' }));
    expect(tiles(out.layout).map((t) => t.tabs[0])).toEqual(['alpha', 'terminal']);
    expect(tiles(out.layout)[0].size).toBeCloseTo(SERVICE_SHARE, 3);
  });
});

describe('openBeside — a new tile right of the target service tile', () => {
  test('splits the target tile; the new tile is focused', () => {
    const layout = row(tile('a', ['alpha']), TERMINAL);
    const out = openBeside(layout, 'beta', ctx({ focusedTile: 'a' }));
    expect(out.changed).toBe(true);
    expect(tiles(out.layout).map((t) => t.tabs[0])).toEqual(['alpha', 'beta', 'terminal']);
    expect(out.focus).toBe(tileOf(out.layout, 'beta')?.tile);
    expect(keysOf(out.layout)).toContain('alpha'); // nothing evicted
  });

  test('an already-docked panel stays where it is', () => {
    const layout = row(tile('a', ['alpha']), tile('b', ['beta']), TERMINAL);
    expect(openBeside(layout, 'alpha', ctx({ focusedTile: 'b' }))).toEqual({
      layout,
      changed: false,
      focus: 'a',
    });
  });

  test('with no service tile it opens the first tile', () => {
    const out = openBeside(TERMINAL, 'alpha', ctx({ focusedTile: 'tt' }));
    expect(tiles(out.layout).map((t) => t.tabs[0])).toEqual(['alpha', 'terminal']);
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
