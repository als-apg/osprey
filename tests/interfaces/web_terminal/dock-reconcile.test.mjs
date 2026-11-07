// @ts-check
// @vitest-environment node
/**
 * Unit tests for the pure `reconcile(layout, registeredKeys)` run on a
 * persisted dock layout before it is applied.
 *
 *   npx vitest run tests/interfaces/web_terminal/dock-reconcile.test.mjs
 *
 * reconcile repairs a dock layout (the dock engine's dock form) against the
 * keys the dock can build: an invalid layout gives null, each unknown key is
 * removed with the engine's removeTab, a tile stored with several panels (by
 * the release that had tabs) keeps the one it showed, and the result is
 * normalised. A layout the dockview-era shell stored (dockview's own
 * serialization) is read into the dock form first. Keys absent from the
 * layout are NOT appended.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe } from 'vitest';

import { reconcile } from '../../../src/osprey/interfaces/web_terminal/static/js/dock-reconcile.js';
import { crowdedTile } from '../../../src/osprey/interfaces/web_terminal/static/js/dock-policy.js';
import { keysOf, normalize, removeTab, validateLayout } from '/static/vendor/dock-engine.mjs';

/**
 * Two services left of the terminal, each in a tile of its own: 0.3 / 0.3 / 0.4.
 * @returns {any}
 */
function servicesAndTerminal() {
  return {
    split: 'row',
    items: [
      { tile: 't1', tabs: ['plots'], showing: 'plots', size: 0.3 },
      { tile: 't3', tabs: ['logbook'], showing: 'logbook', size: 0.3 },
      { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
    ],
  };
}

/**
 * Three tiles: a column of two service tiles beside the terminal.
 * @returns {any}
 */
function threeTiles() {
  return {
    split: 'row',
    items: [
      {
        split: 'column',
        size: 0.6,
        items: [
          { tile: 't1', tabs: ['plots'], showing: 'plots', size: 0.5 },
          { tile: 't3', tabs: ['logbook'], showing: 'logbook', size: 0.5, from: 't1' },
        ],
      },
      { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
    ],
  };
}

/**
 * A dock the release that had tabs stored: a services tile showing logbook
 * in front of plots, left of the terminal at 0.6 / 0.4.
 * @returns {any}
 */
function storedWithTabs() {
  return {
    split: 'row',
    items: [
      { tile: 't1', tabs: ['plots', 'logbook'], showing: 'logbook', size: 0.6 },
      { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
    ],
  };
}

/**
 * The layout the dockview-era shell stored: dockview's own serialization, one
 * group per tile, service placeholders under the `iframe:` prefix.
 * @returns {any}
 */
function dockviewStore() {
  return {
    grid: {
      root: {
        type: 'branch',
        data: [
          {
            type: 'leaf',
            data: { views: ['iframe:plots'], activeView: 'iframe:plots', id: 'g1' },
            size: 600,
          },
          {
            type: 'leaf',
            data: { views: ['terminal'], activeView: 'terminal', id: 'g2' },
            size: 400,
          },
        ],
        size: 800,
      },
      width: 1000,
      height: 800,
      orientation: 'HORIZONTAL',
    },
    panels: {
      'iframe:plots': { id: 'iframe:plots', contentComponent: 'dock-iframe-placeholder' },
      terminal: { id: 'terminal', contentComponent: 'terminal' },
    },
    activeGroup: 'g1',
  };
}

describe('reconcile — clean layout pass-through', () => {
  test('a layout whose keys are all registered comes back normalised and equal', () => {
    const layout = servicesAndTerminal();
    expect(reconcile(layout, ['plots', 'logbook'])).toEqual(normalize(layout));
  });

  test('accepts a raw JSON string as well as a parsed object', () => {
    const layout = threeTiles();
    expect(reconcile(JSON.stringify(layout), ['plots', 'logbook'])).toEqual(
      reconcile(layout, ['plots', 'logbook'])
    );
  });

  test('does not mutate the input layout, even one it migrates', () => {
    for (const layout of [threeTiles(), storedWithTabs()]) {
      const before = JSON.parse(JSON.stringify(layout));
      reconcile(layout, ['plots']);
      reconcile(layout, ['plots', 'logbook']);
      expect(layout).toEqual(before);
    }
  });

  test('the result is a valid dock layout', () => {
    const out = reconcile(threeTiles(), ['logbook']);
    expect(validateLayout(out).valid).toBe(true);
  });

  test('missing sizes are filled in by normalize', () => {
    const layout = {
      split: 'row',
      items: [
        { tile: 't1', tabs: ['plots'], showing: 'plots' },
        { tile: 't2', tabs: ['terminal'], showing: 'terminal' },
      ],
    };
    const out = /** @type {any} */ (reconcile(layout, ['plots']));
    expect(out.items.map((/** @type {any} */ i) => i.size)).toEqual([0.5, 0.5]);
  });
});

describe('reconcile — a tile stored with several panels keeps the one it showed', () => {
  test('the showing panel keeps the tile, its id and size; the others leave the dock', () => {
    expect(reconcile(storedWithTabs(), ['plots', 'logbook'])).toEqual({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['logbook'], showing: 'logbook', size: 0.6 },
        { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
      ],
    });
  });

  test('every such tile of a nested tree is migrated; tiles of one panel are untouched', () => {
    const layout = threeTiles();
    layout.items[0].items[1].tabs = ['logbook', 'alarms'];
    layout.items[0].items[1].showing = 'alarms';
    layout.items[1].tabs = ['terminal', 'ariel'];
    const out = reconcile(layout, ['plots', 'logbook', 'alarms', 'ariel']);
    expect(keysOf(out)).toEqual(['plots', 'alarms', 'terminal']);
    expect(crowdedTile(out)).toBeNull();
    expect(/** @type {any} */ (out).items[0].items[0]).toEqual({
      tile: 't1',
      tabs: ['plots'],
      showing: 'plots',
      size: 0.5,
    });
  });

  test('the terminal stored beside a service keeps the tile only when it was showing', () => {
    const terminalShowing = { tile: 't1', tabs: ['plots', 'terminal'], showing: 'terminal' };
    expect(reconcile(terminalShowing, ['plots'])).toEqual({
      tile: 't1',
      tabs: ['terminal'],
      showing: 'terminal',
    });
    // Behind a service it leaves with the others; the workspace brings it back.
    const serviceShowing = { ...terminalShowing, showing: 'plots' };
    expect(reconcile(serviceShowing, ['plots'])).toEqual({
      tile: 't1',
      tabs: ['plots'],
      showing: 'plots',
    });
  });
});

describe('reconcile — unknown keys are removed with the engine', () => {
  test('an unknown key behind the showing panel is removed; the showing one keeps the tile', () => {
    const out = /** @type {any} */ (reconcile(storedWithTabs(), ['logbook']));
    expect(out.items[0]).toMatchObject({ tile: 't1', tabs: ['logbook'], showing: 'logbook' });
    expect(keysOf(out)).toEqual(['logbook', 'terminal']);
  });

  test('unknown keys go first: an unknown showing panel hands showing on, and the tile keeps that one', () => {
    const layout = {
      tile: 't1',
      tabs: ['plots', 'logbook', 'alarms'],
      showing: 'logbook',
    };
    expect(reconcile(layout, ['plots', 'alarms'])).toEqual({
      tile: 't1',
      tabs: ['alarms'],
      showing: 'alarms',
    });
  });

  test('a tile left with no panel closes by the leave rule', () => {
    const out = reconcile(servicesAndTerminal(), []);
    expect(out).toEqual(
      normalize(
        /** @type {any} */ (
          removeTab(removeTab(servicesAndTerminal(), 'plots').layout, 'logbook').layout
        )
      )
    );
    expect(out).toMatchObject({ tile: 't2', tabs: ['terminal'], showing: 'terminal' });
  });

  test('a closed tile gives its space to its from tile and the split collapses', () => {
    const out = /** @type {any} */ (reconcile(threeTiles(), ['plots']));
    expect(keysOf(out)).toEqual(['plots', 'terminal']);
    expect(out.split).toBe('row');
    expect(out.items[0]).toMatchObject({ tile: 't1', tabs: ['plots'] });
    expect(out.items[0].size).toBeCloseTo(0.6, 3);
    expect(out.items[1].size).toBeCloseTo(0.4, 3);
  });

  test('equals removing every unknown key with removeTab in tree order, then normalize', () => {
    const layout = {
      split: 'row',
      items: [
        { tile: 't1', tabs: ['plots'], showing: 'plots', size: 0.2 },
        { tile: 't3', tabs: ['logbook'], showing: 'logbook', size: 0.2 },
        { tile: 't4', tabs: ['alarms'], showing: 'alarms', size: 0.2 },
        { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
      ],
    };
    /** @type {any} */
    let expected = layout;
    for (const key of ['plots', 'alarms']) expected = removeTab(expected, key).layout;
    expect(reconcile(layout, ['logbook'])).toEqual(normalize(expected));
  });

  test('terminal is always kept, registered or not', () => {
    const out = reconcile(servicesAndTerminal(), ['plots', 'logbook']);
    expect(keysOf(out)).toContain('terminal');
  });

  test('registered keys absent from the layout are not appended', () => {
    const out = reconcile(servicesAndTerminal(), ['plots', 'logbook', 'never-opened']);
    expect(keysOf(out)).not.toContain('never-opened');
  });
});

describe('reconcile — a dockview serialization is read into the dock form', () => {
  test('groups become tiles keyed by stripped ids, with their fractions', () => {
    expect(reconcile(dockviewStore(), ['plots'])).toEqual({
      split: 'row',
      items: [
        { tile: 'g1', tabs: ['plots'], showing: 'plots', size: 0.6 },
        { tile: 'g2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
      ],
    });
  });

  test('as a raw string too, and its unknown keys are removed like any other', () => {
    expect(reconcile(JSON.stringify(dockviewStore()), [])).toEqual({
      tile: 'g2',
      tabs: ['terminal'],
      showing: 'terminal',
    });
  });

  test('a stored group holding several placeholders keeps the one it showed', () => {
    const store = dockviewStore();
    store.grid.root.data[0].data.views = ['iframe:plots', 'iframe:logbook'];
    store.grid.root.data[0].data.activeView = 'iframe:logbook';
    const out = /** @type {any} */ (reconcile(store, ['plots', 'logbook']));
    expect(out.items[0]).toEqual({ tile: 'g1', tabs: ['logbook'], showing: 'logbook', size: 0.6 });
    expect(keysOf(out)).toEqual(['logbook', 'terminal']);
  });

  test('a serialization with no grid tree left is unusable', () => {
    const store = dockviewStore();
    store.grid.root.data = [];
    expect(reconcile(store, ['plots'])).toBeNull();
  });
});

describe('reconcile — registered keys', () => {
  test('any iterable works (a Set)', () => {
    const out = reconcile(servicesAndTerminal(), new Set(['plots']));
    expect(keysOf(out)).toEqual(['plots', 'terminal']);
  });

  test('malformed entries are ignored', () => {
    const out = reconcile(
      servicesAndTerminal(),
      /** @type {any} */ ([null, 42, '', { id: 'logbook' }, 'plots'])
    );
    expect(keysOf(out)).toEqual(['plots', 'terminal']);
  });

  test('a missing list keeps only the terminal', () => {
    const out = reconcile(servicesAndTerminal(), /** @type {any} */ (undefined));
    expect(keysOf(out)).toEqual(['terminal']);
  });
});

describe('reconcile — unusable input falls back (null)', () => {
  test('unparseable JSON string returns null', () => {
    expect(reconcile('{not json', ['plots'])).toBeNull();
  });

  test('null, primitives and non-layout objects return null', () => {
    expect(reconcile(null, ['plots'])).toBeNull();
    expect(reconcile('null', ['plots'])).toBeNull();
    expect(reconcile(42, ['plots'])).toBeNull();
    expect(reconcile({ foo: 'bar' }, ['plots'])).toBeNull();
  });

  test('a layout failing validateLayout returns null', () => {
    expect(reconcile({ tile: 't1', tabs: [], showing: 'plots' }, ['plots'])).toBeNull();
    expect(reconcile({ tile: 't1', tabs: ['plots'], showing: 'logbook' }, ['plots'])).toBeNull();
    expect(
      reconcile({ split: 'row', items: [{ tile: 't1', tabs: ['plots'], showing: 'plots' }] }, [
        'plots',
      ])
    ).toBeNull();
  });

  test('a layout with no registered key left returns null', () => {
    expect(reconcile({ tile: 't1', tabs: ['plots', 'logbook'], showing: 'plots' }, [])).toBeNull();
  });
});
