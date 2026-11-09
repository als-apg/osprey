// @ts-check
/**
 * Contract tests for the sidebar-row drag (rail-drag.js):
 *   npx vitest run tests/interfaces/web_terminal/rail-drag.test.mjs
 *
 * A row is one more drag source of the dock's drop bridge: at drag start it
 * becomes the engine source the bridge reads (dock-tab.js's dockDragSource) —
 * `{new: id}`, or `{tab: id}` when its panel already has a tab in the held
 * layout — and stamps the rail MIME on the transfer; without a dock there is
 * nothing to drop into and the drag is cancelled. A landed drop (the holder's
 * layout gained the key across the document's drop) reports to the landing
 * handler once the drag is over; every end — a drop, a cancel, a lost gesture
 * ended by the document failsafe — clears the dock's source. Placement itself
 * is the bridge's (dock-drop.test.mjs).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const MOD = '../../../src/osprey/interfaces/web_terminal/static/js/rail-drag.js';
const TAB = '../../../src/osprey/interfaces/web_terminal/static/js/dock-tab.js';
const APPLY = '../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js';

const { getDockApi, state } = vi.hoisted(() => ({
  state: { api: /** @type {any} */ (null) },
  getDockApi: vi.fn(() => /** @type {any} */ (null)),
}));
getDockApi.mockImplementation(() => state.api);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi,
  closeDockPanel: vi.fn(),
  closeTerminalPanel: vi.fn(),
  setLayoutRestoredHook: () => {},
  setServiceRedock: () => {},
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/pane-window.js', () => ({
  detach: vi.fn(),
  detachBlocker: () => null,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js', () => ({
  registerContribHost: () => {},
  unregisterContribHost: () => {},
}));

const ONE = { tile: 't1', tabs: ['alpha'], showing: 'alpha' };
const TWO = {
  split: 'row',
  items: [
    { tile: 't1', tabs: ['alpha'], showing: 'alpha', size: 0.5 },
    { tile: 't2', tabs: ['beta', 'terminal'], showing: 'beta', size: 0.5 },
  ],
};

/** A minimal DataTransfer: jsdom has none. */
function transfer() {
  /** @type {Record<string, string>} */
  const store = {};
  return /** @type {any} */ ({
    store,
    setData: (/** @type {string} */ t, /** @type {string} */ v) => {
      store[t] = v;
    },
    getData: (/** @type {string} */ t) => store[t] ?? '',
    effectAllowed: 'all',
  });
}

/** @param {string} type @param {EventTarget} [target] */
function fire(type, target = document) {
  target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true }));
}

const tick = () => new Promise((r) => setTimeout(r, 0));

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  getDockApi.mockImplementation(() => state.api);
  state.api = {};
  document.body.replaceChildren();
});

afterEach(async () => {
  const tab = await import(TAB);
  tab.clearDockDrag();
});

/** A holder over `layout`, with no dockview behind it. @param {any} layout */
async function holderOver(layout) {
  const { createLayoutHolder } = await import(APPLY);
  const holder = createLayoutHolder({ dockPx: { width: 1000, height: 600 } });
  holder.restore(layout);
  return holder;
}

describe('the source a row becomes', () => {
  test('{new} for a panel with no tab, {tab} for one the held layout has', async () => {
    const { railSourceOf } = await import(MOD);
    const holder = await holderOver(ONE);
    expect(railSourceOf('alpha', holder)).toEqual({ tab: 'alpha' });
    expect(railSourceOf('gamma', holder)).toEqual({ new: 'gamma' });
    expect(railSourceOf('gamma', null)).toEqual({ new: 'gamma' });
  });

  test('railDragStart records the source for the bridge, stamps the rail MIME, and reports accepted', async () => {
    const mod = await import(MOD);
    const tab = await import(TAB);
    const holder = await holderOver(TWO);
    mod.initRailDrag({ holder: () => holder });
    const seen = vi.fn();
    tab.onDockDragStart(seen);
    const dt = transfer();

    expect(mod.railDragStart('gamma', dt)).toBe(true);

    expect(mod.railDragSource()).toBe('gamma');
    expect(tab.dockDragSource()).toEqual({ new: 'gamma' });
    expect(seen).toHaveBeenCalledExactlyOnceWith({ new: 'gamma' });
    expect(dt.store[mod.RAIL_DRAG_MIME]).toBe('gamma');
    expect(JSON.parse(dt.store[tab.DOCK_SOURCE_MIME])).toEqual({ new: 'gamma' });
    expect(dt.effectAllowed).toBe('move');
    mod.railDragEnd();
  });

  test("a new tab is built with the row's name", async () => {
    const mod = await import(MOD);
    const holder = await holderOver(TWO);
    mod.initRailDrag({
      holder: () => holder,
      labelOf: (/** @type {string} */ id) => id.toUpperCase(),
    });
    mod.railDragStart('gamma', transfer());
    expect(holder.panelRecord('gamma')).toEqual({ title: 'GAMMA' });
    mod.railDragEnd();
  });

  test('an open panel drags as its tab, so a drop moves it rather than duplicating it', async () => {
    const mod = await import(MOD);
    const tab = await import(TAB);
    const holder = await holderOver(TWO);
    mod.initRailDrag({ holder: () => holder });
    expect(mod.railDragStart('beta', transfer())).toBe(true);
    expect(tab.dockDragSource()).toEqual({ tab: 'beta' });
    mod.railDragEnd();
  });

  test('cancels in fallback mode (no dock to drop into)', async () => {
    state.api = null;
    const mod = await import(MOD);
    const tab = await import(TAB);
    expect(mod.railDragStart('gamma', transfer())).toBe(false);
    expect(mod.railDragSource()).toBeNull();
    expect(tab.dockDragSource()).toBeNull();
  });
});

describe('ending a drag', () => {
  test('railDragEnd forgets the drag and clears the dock source; idempotent', async () => {
    const mod = await import(MOD);
    const tab = await import(TAB);
    mod.initRailDrag({ holder: () => null });
    mod.railDragStart('gamma', transfer());
    mod.railDragEnd();
    mod.railDragEnd();
    expect(mod.railDragSource()).toBeNull();
    expect(tab.dockDragSource()).toBeNull();
  });

  test("a document-level terminator ends the drag when the row's own dragend is lost", async () => {
    const mod = await import(MOD);
    const tab = await import(TAB);
    mod.railDragStart('gamma', transfer());
    fire('pointerup');
    expect(mod.railDragSource()).toBeNull();
    expect(tab.dockDragSource()).toBeNull();
  });

  test('the failsafe disarms with the drag: a later pointerup leaves a newer drag of another source alone', async () => {
    const mod = await import(MOD);
    const tab = await import(TAB);
    mod.railDragStart('gamma', transfer());
    mod.railDragEnd();
    tab.dockDragStart({ tab: 'alpha' }, null);
    fire('pointerup');
    expect(tab.dockDragSource()).toEqual({ tab: 'alpha' });
  });

  test('a drop ends the drag one turn later, after the bridge has read the source', async () => {
    const mod = await import(MOD);
    const tab = await import(TAB);
    const holder = await holderOver(ONE);
    mod.initRailDrag({ holder: () => holder });
    mod.railDragStart('gamma', transfer());
    fire('drop');
    expect(mod.railDragSource()).toBe('gamma');
    await tick();
    expect(mod.railDragSource()).toBeNull();
    expect(tab.dockDragSource()).toBeNull();
  });
});

describe('the landing', () => {
  test('a drop that put the panel in the held layout reports it once the drag is over', async () => {
    const mod = await import(MOD);
    const holder = await holderOver(ONE);
    const onRailDrop = vi.fn();
    mod.initRailDrag({ onRailDrop, holder: () => holder });
    // The bridge's drop handler runs between the landing watch's capture and
    // bubble listeners: a target inside the document that changes the layout.
    const target = document.createElement('div');
    document.body.appendChild(target);
    target.addEventListener('drop', () => {
      holder.restore({
        split: 'row',
        items: [
          { tile: 't1', tabs: ['alpha'], showing: 'alpha', size: 0.5 },
          { tile: 't2', tabs: ['gamma'], showing: 'gamma', size: 0.5 },
        ],
      });
    });
    mod.railDragStart('gamma', transfer());
    fire('drop', target);
    expect(onRailDrop).not.toHaveBeenCalled();
    await tick();
    await tick();
    expect(onRailDrop).toHaveBeenCalledExactlyOnceWith('gamma', { wasOnScreen: false });
  });

  test('a moved tab lands as one that was on screen; a drop that changed nothing lands nowhere', async () => {
    const mod = await import(MOD);
    const holder = await holderOver(TWO);
    const onRailDrop = vi.fn();
    mod.initRailDrag({ onRailDrop, holder: () => holder });
    const target = document.createElement('div');
    document.body.appendChild(target);
    let move = true;
    target.addEventListener('drop', () => {
      if (!move) return;
      holder.restore({
        split: 'row',
        items: [
          { tile: 't2', tabs: ['beta', 'terminal'], showing: 'beta', size: 0.5 },
          { tile: 't1', tabs: ['alpha'], showing: 'alpha', size: 0.5 },
        ],
      });
    });
    mod.railDragStart('beta', transfer());
    fire('drop', target);
    await tick();
    await tick();
    expect(onRailDrop).toHaveBeenCalledExactlyOnceWith('beta', { wasOnScreen: true });

    move = false;
    mod.railDragStart('gamma', transfer());
    fire('drop', target);
    await tick();
    await tick();
    expect(onRailDrop).toHaveBeenCalledTimes(1);
  });

  test('the landing waits for the live drag to end, so the held apply has run', async () => {
    const mod = await import(MOD);
    const holder = await holderOver(ONE);
    const onRailDrop = vi.fn();
    mod.initRailDrag({ onRailDrop, holder: () => holder });
    const target = document.createElement('div');
    document.body.appendChild(target);
    target.addEventListener('drop', () => {
      holder.setDragLive(true);
      holder.restore({ tile: 't1', tabs: ['alpha', 'gamma'], showing: 'gamma' });
    });
    mod.railDragStart('gamma', transfer());
    fire('drop', target);
    await tick();
    await tick();
    expect(onRailDrop).not.toHaveBeenCalled();
    holder.setDragLive(false);
    await tick();
    await tick();
    expect(onRailDrop).toHaveBeenCalledTimes(1);
  });
});
