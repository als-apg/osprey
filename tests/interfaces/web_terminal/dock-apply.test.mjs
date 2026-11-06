// @ts-check
// @vitest-environment jsdom
/**
 * The OSPREY layout holder and its one apply path, against real dockview:
 *   npx vitest run tests/interfaces/web_terminal/dock-apply.test.mjs
 *
 * jsdom with the pinned dockview-core min bundle (see _dock-real.mjs): the
 * serializer's output is only proven by dockview's own fromJSON accepting it
 * and building the groups it describes.
 *
 * Every tile holds one panel (dock-policy.js): a stored dock with a tile of
 * several panels is restored with the panel each such tile showed, and an
 * operation whose result gives a tile several panels is refused by the
 * policy's name and changes nothing.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, describe, expect, test, vi } from 'vitest';

import {
  TAB_KEY_ATTRIBUTE,
  TERMINAL_TITLE,
  createLayoutHolder,
  fromSerialized,
  getLayoutHolder,
  initLayoutHolder,
  isSerializedDockview,
  keyOfPanelId,
  panelIdOf,
  sameShape,
  toSerialized,
} from '../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js';
import {
  closeKey,
  openPanel,
} from '../../../src/osprey/interfaces/web_terminal/static/js/dock-open.js';
import { ONE_PANEL_PER_TILE } from '../../../src/osprey/interfaces/web_terminal/static/js/dock-policy.js';
import {
  MIN_PX,
  closeTab,
  geometry,
  keysOf,
  planDrop,
  tileOf,
} from '/static/vendor/dock-engine.mjs';
import {
  DEFAULT_PX,
  buildSha,
  groupsOf,
  mountDock,
  pinnedDockview,
  settle,
  unmountDocks,
} from './_dock-real.mjs';

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

/** Two services left of the terminal, each in a tile of its own: 0.3 / 0.3 / 0.4. */
/** @type {Layout} */
const THREE_TILES = {
  split: 'row',
  items: [
    { tile: 't1', tabs: ['alpha'], showing: 'alpha', size: 0.3 },
    { tile: 't3', tabs: ['beta'], showing: 'beta', size: 0.3 },
    { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
  ],
};

/** A nested tree: a column of two service tiles beside the terminal. */
/** @type {Layout} */
const NESTED = {
  split: 'row',
  items: [
    {
      split: 'column',
      size: 0.5,
      items: [
        { tile: 't1', tabs: ['alpha'], showing: 'alpha', size: 0.3 },
        { tile: 't3', tabs: ['delta'], showing: 'delta', size: 0.7 },
      ],
    },
    { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.5 },
  ],
};

/**
 * NESTED as the release that had tabs stored it: t3 showed delta in front of
 * gamma, and beta sat behind the terminal in t2.
 * @type {Layout}
 */
const STORED_WITH_TABS = {
  split: 'row',
  items: [
    {
      split: 'column',
      size: 0.5,
      items: [
        { tile: 't1', tabs: ['alpha'], showing: 'alpha', size: 0.3 },
        { tile: 't3', tabs: ['gamma', 'delta'], showing: 'delta', size: 0.7 },
      ],
    },
    { tile: 't2', tabs: ['terminal', 'beta'], showing: 'terminal', size: 0.5 },
  ],
};

/**
 * @param {any} api
 * @param {object} [extra]
 */
function holderFor(api, extra = {}) {
  return createLayoutHolder({ api, dockPx: DEFAULT_PX, ...extra });
}

/**
 * Strip sizes so trees compare by keys and tiles.
 * @param {any} node
 * @returns {any}
 */
function shapeOf(node) {
  if (node === null) return null;
  if ('tile' in node) return { tile: node.tile, tabs: node.tabs, showing: node.showing };
  return { split: node.split, items: node.items.map(shapeOf) };
}

/**
 * Every size in `a` within `eps` of `b`'s (same shape assumed).
 * @param {any} a
 * @param {any} b
 * @param {number} eps
 */
function expectSizesClose(a, b, eps) {
  if (b.size !== undefined) expect(Math.abs((a.size ?? 0) - b.size)).toBeLessThanOrEqual(eps);
  if ('items' in b)
    b.items.forEach((/** @type {any} */ item, /** @type {number} */ i) =>
      expectSizesClose(a.items[i], item, eps)
    );
}

afterEach(() => {
  unmountDocks();
  document.body.innerHTML = '';
});

describe('the dockview devDependency', () => {
  test('node_modules holds the min bundle the vendor manifest pins', () => {
    expect(buildSha(), 'package.json and vendor_manifest.json pin one dockview-core').toBe(
      pinnedDockview().sha256
    );
  });

  test('package.json pins it exactly at the manifest version', () => {
    const pkg = JSON.parse(readFileSync(path.join(REPO_ROOT, 'package.json'), 'utf8'));
    expect(pkg.devDependencies['dockview-core']).toBe(pinnedDockview().version);
  });
});

describe('the serializer', () => {
  test('service keys get the iframe: prefix; terminal keeps its id', () => {
    expect(panelIdOf('alpha')).toBe('iframe:alpha');
    expect(panelIdOf('terminal')).toBe('terminal');
    expect(keyOfPanelId('iframe:alpha')).toBe('alpha');
    expect(keyOfPanelId('terminal')).toBe('terminal');
    // Anything else names no panel this dock builds.
    expect(keyOfPanelId('alpha')).toBeNull();
    expect(keyOfPanelId(undefined)).toBeNull();
  });

  test('group id = tile id, its view = the tile’s panel, active view = showing, 40 px minimums on every panel', () => {
    const s = toSerialized(THREE_TILES, DEFAULT_PX);
    expect(s.grid.orientation).toBe('HORIZONTAL');
    expect(s.grid.root.data.map((/** @type {any} */ n) => n.data)).toEqual([
      { views: ['iframe:alpha'], activeView: 'iframe:alpha', id: 't1' },
      { views: ['iframe:beta'], activeView: 'iframe:beta', id: 't3' },
      { views: ['terminal'], activeView: 'terminal', id: 't2' },
    ]);
    expect(Object.keys(s.panels).sort()).toEqual(['iframe:alpha', 'iframe:beta', 'terminal']);
    for (const entry of Object.values(s.panels)) {
      expect(entry.minimumWidth).toBe(MIN_PX);
      expect(entry.minimumHeight).toBe(MIN_PX);
    }
    expect(s.panels['iframe:alpha'].contentComponent).toBe('dock-iframe-placeholder');
    expect(s.panels.terminal).toMatchObject({ contentComponent: 'terminal', title: 'Agent' });
    expect(TERMINAL_TITLE).toBe('Agent');
  });

  test('sizes are the engine geometry in px', () => {
    const s = toSerialized(THREE_TILES, DEFAULT_PX);
    const geo = geometry(THREE_TILES, { x: 0, y: 0, ...DEFAULT_PX }, MIN_PX);
    const widths = s.grid.root.data.map((/** @type {any} */ n) => n.size);
    expect(widths).toEqual([geo.tiles.t1.width, geo.tiles.t3.width, geo.tiles.t2.width]);
  });

  test('a root tile is wrapped in a branch; null is the empty dock', () => {
    const one = toSerialized({ tile: 't1', tabs: ['terminal'], showing: 'terminal' }, DEFAULT_PX);
    expect(one.grid.root.type).toBe('branch');
    expect(one.grid.root.data[0].data.id).toBe('t1');
    const none = toSerialized(null, DEFAULT_PX);
    expect(none.grid.root).toEqual({ type: 'branch', data: [] });
    expect(none.panels).toEqual({});
  });

  test('a stored panel entry keeps its state; the input is never mutated', () => {
    const input = structuredClone(THREE_TILES);
    const s = toSerialized(input, DEFAULT_PX, {
      panels: {
        'iframe:alpha': {
          id: 'iframe:alpha',
          contentComponent: 'dock-iframe-placeholder',
          title: 'Alpha',
          params: { x: 1 },
        },
      },
      activeTile: 't2',
    });
    expect(s.panels['iframe:alpha']).toMatchObject({
      title: 'Alpha',
      params: { x: 1 },
      minimumWidth: MIN_PX,
    });
    expect(s.activeGroup).toBe('t2');
    expect(input).toEqual(THREE_TILES);
  });

  test('a stored terminal entry carrying an older title is healed to the Agent title', () => {
    const s = toSerialized(THREE_TILES, DEFAULT_PX, {
      panels: { terminal: { id: 'terminal', contentComponent: 'terminal', title: 'SESSION' } },
    });
    expect(s.panels.terminal.title).toBe('Agent');
  });

  test('a record stands in for a key with no stored entry', () => {
    const s = toSerialized(THREE_TILES, DEFAULT_PX, {
      newPanel: (key) => (key === 'beta' ? { title: 'Beta', params: { y: 2 } } : undefined),
    });
    expect(s.panels['iframe:beta']).toMatchObject({ title: 'Beta', params: { y: 2 } });
    expect(s.panels['iframe:alpha'].title).toBe('alpha');
  });

  test.each([
    ['three tiles', THREE_TILES],
    ['a nested tree', NESTED],
    ['a root tile', { tile: 't1', tabs: ['alpha'], showing: 'alpha' }],
  ])('keys round-trip through dockview: %s', async (_name, layout) => {
    const { api } = mountDock();
    api.fromJSON(toSerialized(/** @type {Layout} */ (layout), DEFAULT_PX));
    await settle();
    const back = fromSerialized(api.toJSON());
    expect(shapeOf(back)).toEqual(shapeOf(layout));
    expectSizesClose(back, layout, 0.005);
    // Every dockview id the dock holds is a prefixed service or the terminal.
    for (const id of api.panels.map((/** @type {any} */ p) => p.id)) {
      expect(id === 'terminal' || id.startsWith('iframe:')).toBe(true);
    }
  });

  test('fromSerialized of an empty grid is null', () => {
    expect(fromSerialized(toSerialized(null, DEFAULT_PX))).toBeNull();
    expect(fromSerialized(undefined)).toBeNull();
  });

  test('fromSerialized reads a stored dockview layout and drops ids the dock never builds', () => {
    // The shape the dockview-era shell stored: a leaf per group, an unprefixed
    // id for a panel that is not a service placeholder, a dangling activeGroup.
    const stored = {
      grid: {
        root: {
          type: 'branch',
          data: [
            {
              type: 'leaf',
              data: { views: ['iframe:alpha', 'stray'], activeView: 'iframe:alpha', id: 'g1' },
              size: 600,
            },
            {
              type: 'leaf',
              data: { views: ['terminal'], activeView: 'terminal', id: 'g2' },
              size: 400,
            },
          ],
        },
        width: 1000,
        height: 600,
        orientation: 'HORIZONTAL',
      },
      panels: {},
      activeGroup: 'gone',
    };
    expect(isSerializedDockview(stored)).toBe(true);
    expect(isSerializedDockview(THREE_TILES)).toBe(false);
    expect(fromSerialized(stored)).toEqual({
      split: 'row',
      items: [
        { tile: 'g1', tabs: ['alpha'], showing: 'alpha', size: 0.6 },
        { tile: 'g2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
      ],
    });
  });

  test('fromSerialized refuses a tile id used twice', () => {
    const twice = {
      grid: {
        root: {
          type: 'branch',
          data: [
            {
              type: 'leaf',
              data: { views: ['iframe:alpha'], activeView: 'iframe:alpha', id: 'g' },
            },
            { type: 'leaf', data: { views: ['terminal'], activeView: 'terminal', id: 'g' } },
          ],
        },
        orientation: 'HORIZONTAL',
      },
      panels: {},
    };
    expect(fromSerialized(twice)).toBeNull();
  });
});

describe('applying the held layout', () => {
  test('restore builds one group per tile, each holding its one panel', async () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    expect(holder.restore(NESTED)).toBe(true);
    await settle();
    expect(groupsOf(api)).toEqual(
      expect.arrayContaining([
        { tile: 't1', views: ['iframe:alpha'], active: 'iframe:alpha' },
        { tile: 't3', views: ['iframe:delta'], active: 'iframe:delta' },
        { tile: 't2', views: ['terminal'], active: 'terminal' },
      ])
    );
    expect(api.groups).toHaveLength(3);
  });

  test('restore keeps the panel a stored tile of several showed; the others leave the dock', async () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    expect(holder.restore(STORED_WITH_TABS)).toBe(true);
    // The same tiles at the same sizes, each with the panel it showed.
    expect(holder.layout).toEqual(NESTED);
    expect(keysOf(holder.layout)).toEqual(['alpha', 'delta', 'terminal']);
    await settle();
    expect(groupsOf(api)).toEqual(
      expect.arrayContaining([
        { tile: 't1', views: ['iframe:alpha'], active: 'iframe:alpha' },
        { tile: 't3', views: ['iframe:delta'], active: 'iframe:delta' },
        { tile: 't2', views: ['terminal'], active: 'terminal' },
      ])
    );
    expect(api.groups).toHaveLength(3);
    expect(api.getPanel('iframe:gamma')).toBeFalsy();
    expect(api.getPanel('iframe:beta')).toBeFalsy();
  });

  test('an apply followed by a microtask writes nothing to the holder', async () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    holder.restore(THREE_TILES);
    await settle();
    holder.ack(7);
    let layoutEvents = 0;
    const sub = api.onDidLayoutChange(() => layoutEvents++);
    const before = structuredClone({
      layout: holder.layout,
      focusedTile: holder.focusedTile,
      lastServiceTile: holder.lastServiceTile,
      closed: holder.closed,
    });
    const applied = holder.apply();
    await Promise.resolve();
    await settle();
    await applied;
    sub.dispose();
    expect(layoutEvents).toBeGreaterThan(0);
    expect({
      layout: holder.layout,
      focusedTile: holder.focusedTile,
      lastServiceTile: holder.lastServiceTile,
      closed: holder.closed,
    }).toEqual(before);
    expect(holder.dirty).toBe(false);
    expect(holder.ackedRev).toBe(7);
  });

  test('show records a click in a tile without re-applying; the next open lands in that tile', async () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    holder.restore(THREE_TILES);
    await settle();
    holder.ack(1);
    const layout = holder.layout;
    const fromJSON = vi.spyOn(api, 'fromJSON');

    // A person clicks in beta's tile: dockview focuses it, the bar tells the holder.
    api.getPanel('iframe:beta').api.setActive();
    expect(holder.show('t3', 'beta')).toBe(true);
    expect(holder.focusedTile).toBe('t3');
    expect(holder.lastServiceTile).toBe('t3');
    expect(holder.layout).toBe(layout);
    expect(holder.dirty).toBe(false);
    expect(fromJSON).not.toHaveBeenCalled();

    // An unknown tile, or a tile not holding the key, is refused.
    expect(holder.show('t9', 'beta')).toBe(false);
    expect(holder.show('t1', 'beta')).toBe(false);
    expect(holder.focusedTile).toBe('t3');

    holder.run((current, ctx) => openPanel(current, 'gamma', ctx));
    expect(tileOf(holder.layout, 'gamma')).toMatchObject({ tile: 't3', tabs: ['gamma'] });
    expect(tileOf(holder.layout, 'alpha')?.tile).toBe('t1');
  });

  test('a result giving a tile several panels is refused by name; nothing changes', async () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    holder.restore(THREE_TILES);
    await settle();
    holder.ack(4);
    const layout = holder.layout;
    const fromJSON = vi.spyOn(api, 'fromJSON');

    // The engine's centre drop puts gamma into t1 beside alpha, as the release
    // with tabs did.
    const result = holder.run((current, ctx) => ({
      ...planDrop(current, { new: 'gamma' }, { tile: 't1', zone: 'centre' }, ctx.dock),
      focus: 't1',
      record: { kind: 'layout', layout: current },
    }));

    expect(result).toEqual({ layout, changed: false, refusal: ONE_PANEL_PER_TILE });
    expect(holder.layout).toBe(layout);
    expect(holder.dirty).toBe(false);
    expect(holder.closed).toHaveLength(0);
    expect(holder.focusedTile).toBeNull();
    await settle();
    expect(fromJSON).not.toHaveBeenCalled();
    expect(api.getPanel('iframe:gamma')).toBeFalsy();
    expect(api.groups.map((/** @type {any} */ g) => g.panels.length)).toEqual([1, 1, 1]);
  });

  test('an injected guard wraps fromJSON', () => {
    const { api } = mountDock();
    /** @type {string[]} */
    const calls = [];
    const fromJSON = api.fromJSON.bind(api);
    api.fromJSON = (/** @type {any} */ s) => {
      calls.push('fromJSON');
      fromJSON(s);
    };
    const guard = vi.fn((/** @type {() => any} */ fn) => {
      calls.push('enter');
      const out = fn();
      calls.push('leave');
      return out;
    });
    createLayoutHolder({ api, guard, dockPx: DEFAULT_PX }).restore(THREE_TILES);
    expect(calls).toEqual(['enter', 'fromJSON', 'leave']);
  });

  test('a failing fromJSON is logged, not thrown', async () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    api.fromJSON = () => {
      throw new Error('boom');
    };
    expect(holder.restore(THREE_TILES)).toBe(true);
    expect(await holder.apply()).toBe(false);
    expect(error).toHaveBeenCalledWith('dock: fromJSON failed', expect.any(Error));
    error.mockRestore();
  });

  test('applies are held while a drag is live and run once after it ends', async () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    holder.restore(THREE_TILES);
    await settle();
    const fromJSON = vi.spyOn(api, 'fromJSON');
    holder.setDragLive(true);
    holder.run((layout, ctx) =>
      planDrop(layout, { new: 'gamma' }, { tile: 't1', edge: 'top' }, ctx.dock)
    );
    holder.run((layout) => closeTab(layout, 'beta'));
    expect(fromJSON).not.toHaveBeenCalled();
    // The holder already reflects both operations for hit testing.
    expect(holder.layout).not.toBeNull();
    holder.setDragLive(false);
    expect(fromJSON).not.toHaveBeenCalled();
    await settle();
    expect(fromJSON).toHaveBeenCalledTimes(1);
    expect(api.getPanel('iframe:gamma')).toBeTruthy();
    expect(api.getPanel('iframe:beta')).toBeFalsy();
  });

  test('a record set on the holder names a restored panel', async () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    holder.setPanelRecord('alpha', { title: 'Alpha' });
    holder.restore(THREE_TILES);
    await settle();
    expect(api.getPanel('iframe:alpha').title).toBe('Alpha');
    expect(holder.panelRecord('alpha')).toEqual({ title: 'Alpha' });
    expect(holder.panelRecord('beta')).toBeUndefined();
  });
});

describe('the terminal, the overlays and focus across an apply', () => {
  test('the terminal subtree is relocated, not rebuilt, and the xterm keeps focus', async () => {
    const { api, terminal, xterm } = mountDock();
    const holder = holderFor(api);
    holder.restore(THREE_TILES);
    await settle();
    xterm.focus();
    expect(document.activeElement).toBe(xterm);

    // Move the terminal to the other side.
    const fromJSON = vi.spyOn(api, 'fromJSON');
    const result = holder.run((layout, ctx) =>
      planDrop(layout, { tile: 't2' }, { outer: 'left' }, ctx.dock)
    );
    expect(result.changed).toBe(true);
    expect(fromJSON).toHaveBeenCalledTimes(1);
    await settle();
    expect(api.groups.map((/** @type {any} */ g) => g.id)).toEqual(['t2', 't1', 't3']);
    expect(terminal.isConnected).toBe(true);
    const group = api.groups.find((/** @type {any} */ g) => g.id === 't2');
    expect(group.element.contains(terminal)).toBe(true);
    expect(document.activeElement).toBe(xterm);
  });

  test('an iframe overlay outside the dock root survives and its placeholder is re-found', async () => {
    const { api } = mountDock();
    const overlay = document.createElement('div');
    overlay.className = 'dock-iframe-overlay';
    const iframe = document.createElement('iframe');
    overlay.appendChild(iframe);
    document.body.appendChild(overlay);
    const holder = holderFor(api);
    holder.restore(THREE_TILES);
    await settle();
    holder.run((layout, ctx) =>
      planDrop(layout, { tab: 'beta' }, { tile: 't2', edge: 'top' }, ctx.dock)
    );
    await settle();
    expect(iframe.isConnected).toBe(true);
    expect(api.getPanel('iframe:beta')).toBeTruthy();
  });

  /**
   * Stand-in header bars (the dock-tab bar): a focusable element per panel,
   * stamped with its key, put back into each rebuilt group by a layout
   * listener, as dockview re-renders the real bars.
   * @param {any} api
   */
  function standInBars(api) {
    /** @param {string} tile @param {string} key @returns {HTMLElement | null} */
    const bar = (tile, key) => {
      const group = api.groups.find((/** @type {any} */ g) => g.id === tile);
      return group?.element.querySelector(`[${TAB_KEY_ATTRIBUTE}="${key}"]`) ?? null;
    };
    const addBars = () => {
      for (const group of api.groups) {
        for (const panel of group.panels) {
          const key = String(keyOfPanelId(panel.id));
          if (bar(group.id, key)) continue;
          const el = document.createElement('button');
          el.setAttribute(TAB_KEY_ATTRIBUTE, key);
          group.element.appendChild(el);
        }
      }
    };
    addBars();
    const sub = api.onDidLayoutChange(addBars);
    return { bar, addBars, dispose: () => sub.dispose() };
  }

  test('a focused header bar is re-found by tile and key after the rebuild', async () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    holder.restore(THREE_TILES);
    await settle();
    const bars = standInBars(api);
    bars.bar('t3', 'beta')?.focus();
    expect(document.activeElement?.getAttribute(TAB_KEY_ATTRIBUTE)).toBe('beta');
    const restored = holder.apply();
    await settle();
    bars.addBars();
    expect(await restored).toBe(true);
    bars.dispose();
    expect(document.activeElement).toBe(bars.bar('t3', 'beta'));
  });

  test('a focused bar whose panel was replaced goes to the bar of the panel its tile now shows', async () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    holder.restore(THREE_TILES);
    await settle();
    const bars = standInBars(api);
    holder.show('t3', 'beta');
    bars.bar('t3', 'beta')?.focus();

    // An open replaces beta in the focused tile t3; beta leaves the dock.
    holder.run((layout, ctx) => openPanel(layout, 'gamma', ctx));
    for (let turn = 0; turn < 4; turn++) await settle();
    bars.dispose();

    expect(tileOf(holder.layout, 'beta')).toBeNull();
    expect(document.activeElement).not.toBeNull();
    expect(document.activeElement).toBe(bars.bar('t3', 'gamma'));
  });
});

describe('the holder state', () => {
  test('engine records are kept for undo and reopenLast restores the tile', () => {
    const holder = createLayoutHolder({ dockPx: DEFAULT_PX });
    holder.restore(THREE_TILES);
    expect(holder.closed).toHaveLength(0);
    holder.run((layout) => closeTab(layout, 'beta'));
    expect(holder.closed).toHaveLength(1);
    expect(keysOf(holder.layout)).toEqual(['alpha', 'terminal']);
    expect(holder.reopenLast()).toBe(true);
    expect(holder.closed).toHaveLength(0);
    // The tile comes back where it was, at its old share.
    expect(shapeOf(holder.layout)).toEqual(shapeOf(THREE_TILES));
    expectSizesClose(holder.layout, THREE_TILES, 0.005);
    expect(holder.reopenLast()).toBe(false);
  });

  test('reopenLast of a record that would put a panel into an occupied tile is refused', () => {
    const holder = createLayoutHolder({ dockPx: DEFAULT_PX });
    holder.restore(THREE_TILES);
    // A tab record from the release that had tabs: gamma was t1's second tab,
    // and t1 shows alpha now.
    holder.run((layout) => ({
      layout,
      changed: false,
      record: { kind: 'tab', key: 'gamma', tile: 't1', index: 1, showing: true },
    }));
    holder.ack(2);
    const layout = holder.layout;
    expect(holder.reopenLast()).toBe(false);
    expect(holder.layout).toBe(layout);
    expect(holder.dirty).toBe(false);

    // A closed tile whose neighbour has left too: the engine's fallback would
    // put alpha into the terminal's tile, beside the panel it shows.
    holder.restore(THREE_TILES);
    holder.run((current) => closeTab(current, 'alpha'));
    holder.run((current) => closeKey(current, 'beta'));
    holder.ack(3);
    const alone = holder.layout;
    expect(alone).toEqual({ tile: 't2', tabs: ['terminal'], showing: 'terminal' });
    expect(holder.reopenLast()).toBe(false);
    expect(holder.layout).toBe(alone);
    expect(holder.dirty).toBe(false);
  });

  test('restore refuses an invalid layout and leaves the holder untouched', () => {
    const holder = createLayoutHolder({ dockPx: DEFAULT_PX });
    holder.restore(THREE_TILES);
    const before = holder.layout;
    expect(holder.restore({ tile: 't1', tabs: [], showing: 'x' })).toBe(false);
    expect(holder.restore({ split: 'row', items: [] })).toBe(false);
    expect(holder.layout).toBe(before);
  });

  test('restore does not keep the caller object', () => {
    const holder = createLayoutHolder({ dockPx: DEFAULT_PX });
    const input = structuredClone(THREE_TILES);
    holder.restore(input);
    /** @type {any} */ (input).items[0].tabs.push('zeta');
    /** @type {any} */ (input).items[0].showing = 'zeta';
    expect(holder.layout && 'items' in holder.layout && holder.layout.items[0]).toMatchObject({
      tabs: ['alpha'],
      showing: 'alpha',
    });
  });

  test('resized takes sizes from a same-shape tree only and never re-applies', () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    holder.restore(THREE_TILES);
    holder.ack(1);
    const fromJSON = vi.spyOn(api, 'fromJSON');
    const sized = structuredClone(THREE_TILES);
    /** @type {any} */ (sized).items[0].size = 0.2;
    /** @type {any} */ (sized).items[1].size = 0.3;
    /** @type {any} */ (sized).items[2].size = 0.5;
    expect(holder.resized(sized, { width: 1200, height: 700 })).toBe(true);
    expect(
      holder.layout && 'items' in holder.layout && holder.layout.items.map((i) => i.size)
    ).toEqual([0.2, 0.3, 0.5]);
    expect(holder.dock().px).toEqual({ width: 1200, height: 700 });
    expect(holder.dirty).toBe(true);
    const other = { tile: 't1', tabs: ['alpha'], showing: 'alpha' };
    expect(holder.resized(other)).toBe(false);
    expect(fromJSON).not.toHaveBeenCalled();
  });

  test('the same sizes read back are not a change', () => {
    const holder = createLayoutHolder({ dockPx: DEFAULT_PX });
    holder.restore(THREE_TILES);
    holder.ack(1);
    expect(holder.resized(structuredClone(THREE_TILES))).toBe(true);
    expect(holder.dirty).toBe(false);
  });

  test('sizes read back from dockview after a sash drag feed resized', async () => {
    const { api } = mountDock();
    const holder = holderFor(api);
    holder.restore(THREE_TILES);
    await settle();
    const back = fromSerialized(api.toJSON());
    expect(sameShape(holder.layout, back)).toBe(true);
    expect(holder.resized(back)).toBe(true);
  });

  test('focused and last-service tiles follow show, focus and closes', () => {
    const holder = createLayoutHolder({ dockPx: DEFAULT_PX });
    holder.restore(THREE_TILES);
    expect(holder.focusedTile).toBeNull();
    holder.show('t2', 'terminal');
    expect(holder.focusedTile).toBe('t2');
    expect(holder.lastServiceTile).toBeNull();
    holder.show('t3', 'beta');
    expect(holder.lastServiceTile).toBe('t3');
    holder.show('t2', 'terminal');
    expect(holder.lastServiceTile).toBe('t3');
    holder.run((layout) => closeTab(closeTab(layout, 'alpha').layout, 'beta'));
    expect(holder.lastServiceTile).toBeNull();
    expect(holder.focusedTile).toBe('t2');
  });

  test('pop-outs are named beside the tiles and bump the layout', () => {
    const holder = createLayoutHolder({ dockPx: DEFAULT_PX });
    holder.restore(THREE_TILES);
    holder.ack(1);
    expect(holder.popouts).toEqual([]);
    expect(holder.setPoppedOut('beta', true)).toBe(true);
    expect(holder.setPoppedOut('beta', true)).toBe(false);
    expect(holder.setPoppedOut('alpha', true)).toBe(true);
    expect(holder.popouts).toEqual(['alpha', 'beta']);
    expect(holder.dirty).toBe(true);
    holder.ack(2);
    expect(holder.setPoppedOut('beta', false)).toBe(true);
    expect(holder.popouts).toEqual(['alpha']);
    expect(holder.dirty).toBe(true);
    expect(holder.setPoppedOut('beta', false)).toBe(false);
  });

  test('a change marks the holder dirty and ack clears it', () => {
    const holder = createLayoutHolder({ dockPx: DEFAULT_PX });
    expect(holder.dirty).toBe(false);
    holder.restore(THREE_TILES);
    expect(holder.dirty).toBe(true);
    holder.ack(3);
    expect(holder.dirty).toBe(false);
    expect(holder.ackedRev).toBe(3);
    // An operation that changes nothing leaves it clean.
    holder.run((layout) => ({ layout, changed: false }));
    expect(holder.dirty).toBe(false);
  });

  test('initLayoutHolder installs the page holder', () => {
    expect(getLayoutHolder()).toBeNull();
    const holder = initLayoutHolder({ dockPx: DEFAULT_PX });
    expect(getLayoutHolder()).toBe(holder);
  });
});
