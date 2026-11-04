// @ts-check
// @vitest-environment jsdom
/**
 * The dock drop bridge (dock-drop.js) as the single decider, against the pinned
 * real dockview-core in jsdom, the real layout holder, the real header bars and
 * real DOM drag events — never a hand-raised shield:
 *   npx vitest run tests/interfaces/web_terminal/dock-drop.test.mjs
 *
 * Pinned here: a foreign drag gets nothing; a recorded source over a tile edge
 * paints a split preview and raises the shield, and its drop runs the engine
 * through the holder once the drag has ended; a sidebar row's drop onto a
 * header or a centre has no target (a row makes a new tile) while its edges
 * split; the Agent pane's tile takes no tab from any source and never tabs into
 * another tile, while its edges still split; dockview's own tab drag is the
 * tile, its overlays never paint and it moves nothing by itself; a drag that
 * leaves the dock leaves no highlight; Escape ends the drag.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import { loadDockview, settle } from './_dock-real.mjs';

const h = vi.hoisted(() => ({ shield: /** @type {boolean[]} */ ([]) }));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi: () => null,
  closeDockPanel: vi.fn(),
  closeTerminalPanel: vi.fn(),
  setLayoutRestoredHook: () => {},
  setServiceRedock: () => {},
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/popout.js', () => ({
  popOutPanel: vi.fn(),
  canPopOut: () => true,
  isPoppedOut: () => false,
  onPopoutChange: () => {},
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js', () => ({
  registerContribHost: () => {},
  unregisterContribHost: () => {},
}));

import {
  DRAGGING_CLASS,
  PREVIEW_CLASS,
  admitted,
  initDockDrop,
  previewKind,
  setDragShield,
} from '../../../src/osprey/interfaces/web_terminal/static/js/dock-drop.js';
import { createLayoutHolder } from '../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js';
import {
  OSPREY_TAB_COMPONENT,
  clearDockDrag,
  createTileTab,
  dockDragSource,
  dockDragStart,
} from '../../../src/osprey/interfaces/web_terminal/static/js/dock-tab.js';

/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */

const W = 1000;
const H = 600;

/** Tiles t1 | t2, half each: a service tile and the Agent pane's. */
/** @type {Layout} */
const AB = {
  split: 'row',
  items: [
    { tile: 't1', tabs: ['alpha'], showing: 'alpha', size: 0.5 },
    { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.5 },
  ],
};

/** Inside t1's content, in its bottom edge band (31% of 600 from the bottom, above the 48 px dock strip). */
const T1_BOTTOM_EDGE = { clientX: 250, clientY: 500 };
/** t1's centre. */
const T1_CENTRE = { clientX: 250, clientY: 300 };
/** t2's centre: the Agent pane. */
const T2_CENTRE = { clientX: 750, clientY: 300 };
/** Inside t2's content, in its bottom edge band. */
const T2_BOTTOM_EDGE = { clientX: 750, clientY: 500 };

const sizeProps = /** @type {const} */ (['offsetWidth', 'offsetHeight']);

/** @type {Array<() => void>} */
const teardowns = [];

beforeEach(() => {
  h.shield.length = 0;
  setDragShield((on) => h.shield.push(on));
  // jsdom has no PointerEvent; dockview's drag source checks for it on dragstart.
  vi.stubGlobal('PointerEvent', class extends MouseEvent {});
  for (const prop of sizeProps) {
    Object.defineProperty(HTMLElement.prototype, prop, { configurable: true, get: () => 300 });
  }
});

afterEach(() => {
  clearDockDrag();
  for (const teardown of teardowns.splice(0)) teardown();
  for (const prop of sizeProps) delete (/** @type {any} */ (HTMLElement.prototype)[prop]);
  vi.unstubAllGlobals();
  setDragShield(null);
  document.body.replaceChildren();
});

/**
 * Mount a real dockview the way dock-workspace.js does (header bars, html5
 * drags), a real holder showing `layout`, and the bridge over both.
 * @param {Layout} [layout]
 */
function mount(layout = AB) {
  const dockview = loadDockview();
  const container = document.createElement('div');
  container.className = 'main-container';
  document.body.appendChild(container);
  const root = document.createElement('div');
  root.id = 'dock-root';
  root.getBoundingClientRect = () => new DOMRect(0, 0, W, H);
  container.appendChild(root);
  const api = dockview.createDockview(root, {
    disableFloatingGroups: true,
    createComponent: () => ({ element: document.createElement('div'), init() {} }),
    createTabComponent: (/** @type {{ id: string }} */ o) => createTileTab(o.id),
    defaultTabComponent: OSPREY_TAB_COMPONENT,
    dndStrategy: 'html5',
  });
  api.layout(W, H);
  const holder = createLayoutHolder({ api, guard: (fn) => fn(), dockPx: { width: W, height: H } });
  holder.restore(layout);
  const run = vi.spyOn(holder, 'run');
  const dispose = initDockDrop(api, root, { holder: () => holder });
  teardowns.push(() => {
    dispose();
    api.dispose();
  });
  return { api, root, container, holder, run, dispose };
}

/** @typedef {ReturnType<typeof mount>} Mounted */

/** A minimal DataTransfer: jsdom has none. */
function transfer() {
  /** @type {Record<string, string>} */
  const store = {};
  return /** @type {any} */ ({
    get types() {
      return Object.keys(store);
    },
    getData: (/** @type {string} */ type) => store[type] ?? '',
    setData: (/** @type {string} */ type, /** @type {string} */ value) => {
      store[type] = value;
    },
    clearData: () => {},
    setDragImage: () => {},
    effectAllowed: 'all',
    dropEffect: 'none',
    items: { length: 0 },
    files: { length: 0 },
  });
}

/**
 * @param {Element} target
 * @param {string} type
 * @param {DataTransfer} dataTransfer
 * @param {{ clientX: number, clientY: number }} [point]
 * @returns {MouseEvent}
 */
function fire(target, type, dataTransfer, point = T1_CENTRE) {
  const event = new MouseEvent(type, { bubbles: true, cancelable: true, ...point });
  Object.defineProperty(event, 'dataTransfer', { value: dataTransfer });
  target.dispatchEvent(event);
  return event;
}

/** @param {Mounted} m @param {string} tile @returns {HTMLElement} */
function groupOf(m, tile) {
  const group = m.api.groups.find((/** @type {any} */ g) => g.id === tile);
  if (!group) throw new Error(`no group ${tile}`);
  return group.element;
}

/** @param {Mounted} m @param {string} tile @returns {HTMLElement} */
function dvTab(m, tile) {
  return /** @type {HTMLElement} */ (groupOf(m, tile).querySelector('.dv-tab'));
}

/** @param {Mounted} m @param {string} tile @returns {HTMLElement} */
function content(m, tile) {
  return /** @type {HTMLElement} */ (groupOf(m, tile).querySelector('.dv-content-container'));
}

/** Every overlay any group or the api would show, recorded after the bridge ran. @param {Mounted} m */
function overlays(m) {
  /** @type {{ defaultPrevented: boolean }[]} */
  const seen = [];
  for (const group of m.api.groups)
    group.model.onWillShowOverlay((/** @type {any} */ e) => seen.push(e));
  m.api.onWillShowOverlay((/** @type {any} */ e) => seen.push(e));
  return seen;
}

/** @param {Mounted} m */
function previewRects(m) {
  return [...m.container.querySelectorAll(`.${PREVIEW_CLASS}__rect`)].map((el) => el.className);
}

/** @param {Mounted} m */
const dragging = (m) => m.container.classList.contains(DRAGGING_CLASS);

/** The holder's tiles' tabs, in tree order. @param {any} holder */
function tabs(holder) {
  return shape(holder).map((t) => t.slice(t.indexOf(':') + 1));
}

/** The holder's tiles as `tile:tabs`. @param {any} holder */
function shape(holder) {
  /** @type {string[]} */
  const out = [];
  const walk = (/** @type {any} */ n) => {
    if (!n) return;
    if ('tabs' in n) out.push(`${n.tile}:${n.tabs.join('+')}`);
    else n.items.forEach(walk);
  };
  walk(holder.layout);
  return out;
}

describe('mount', () => {
  test('the holder layout is on the dock: group id = tile id, each with its header bar', () => {
    const m = mount();
    expect(m.api.groups.map((/** @type {any} */ g) => g.id)).toEqual(['t1', 't2']);
    expect(dvTab(m, 't1').querySelector('.tile-tab[data-tile="t1"]')).not.toBeNull();
  });
});

describe('a foreign drag', () => {
  test('gets nothing: no preview, no shield, no drop', () => {
    const m = mount();
    fire(content(m, 't1'), 'dragover', transfer(), T1_BOTTOM_EDGE);
    expect(previewRects(m)).toEqual([]);
    expect(dragging(m)).toBe(false);
    expect(h.shield).toEqual([]);
    fire(content(m, 't1'), 'drop', transfer(), T1_BOTTOM_EDGE);
    expect(m.run).not.toHaveBeenCalled();
  });

  test('a source the layout does not know is foreign too', () => {
    const m = mount();
    const dt = transfer();
    dockDragStart({ tab: 'nowhere' }, dt);
    fire(content(m, 't1'), 'dragover', dt, T1_BOTTOM_EDGE);
    expect(previewRects(m)).toEqual([]);
    clearDockDrag();
    dockDragStart({ tile: 't9' }, dt);
    fire(content(m, 't1'), 'dragover', dt, T1_BOTTOM_EDGE);
    expect(previewRects(m)).toEqual([]);
    fire(content(m, 't1'), 'drop', dt, T1_BOTTOM_EDGE);
    expect(m.run).not.toHaveBeenCalled();
  });
});

describe('a recorded source', () => {
  test('over a tile edge: prevented, a split preview, the shield up; its drop runs the engine once the drag ends', async () => {
    const m = mount();
    const dt = transfer();
    dockDragStart({ new: 'gamma' }, dt);
    expect(dragging(m)).toBe(true);
    expect(h.shield).toEqual([true]);

    const over = fire(content(m, 't1'), 'dragover', dt, T1_BOTTOM_EDGE);
    expect(over.defaultPrevented).toBe(true);
    expect(/** @type {any} */ (over).dataTransfer.dropEffect).toBe('move');
    expect(previewRects(m)).toEqual([`${PREVIEW_CLASS}__rect ${PREVIEW_CLASS}__rect--split`]);
    expect(m.holder.dragLive).toBe(true);

    fire(content(m, 't1'), 'drop', dt, T1_BOTTOM_EDGE);
    expect(m.run).toHaveBeenCalledTimes(1);
    expect(shape(m.holder)).toEqual(['t1:alpha', 't3:gamma', 't2:terminal']);
    expect(previewRects(m)).toEqual([]);
    // The apply is held until the drag has ended: the end lands a turn after
    // the drop, and the held apply a turn after that.
    await settle();
    await settle();
    expect(m.holder.dragLive).toBe(false);
    expect(dragging(m)).toBe(false);
    expect(h.shield).toEqual([true, false]);
    expect(dockDragSource()).toBeNull();
    expect(m.api.groups.map((/** @type {any} */ g) => g.id).sort()).toEqual(['t1', 't2', 't3']);
  });

  test('a sidebar row has no target on a tile centre or header: a row makes a new tile', () => {
    const m = mount();
    const dt = transfer();
    dockDragStart({ new: 'gamma' }, dt);
    const over = fire(content(m, 't1'), 'dragover', dt, T1_CENTRE);
    expect(over.defaultPrevented).toBe(true);
    expect(previewRects(m)).toEqual([]);
    fire(content(m, 't1'), 'drop', dt, T1_CENTRE);
    expect(m.run).not.toHaveBeenCalled();
    expect(shape(m.holder)).toEqual(['t1:alpha', 't2:terminal']);
  });

  test('an open tab over another service tile centre is a tab of it', () => {
    const m = mount({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['alpha'], showing: 'alpha', size: 0.5 },
        { tile: 't2', tabs: ['beta'], showing: 'beta', size: 0.5 },
      ],
    });
    const dt = transfer();
    dockDragStart({ tab: 'alpha' }, dt);
    fire(content(m, 't2'), 'dragover', dt, T2_CENTRE);
    expect(previewRects(m)).toEqual([`${PREVIEW_CLASS}__rect ${PREVIEW_CLASS}__rect--tab`]);
    fire(content(m, 't2'), 'drop', dt, T2_CENTRE);
    expect(shape(m.holder)).toEqual(['t2:beta+alpha']);
  });
});

describe('the Agent pane is always its own tile', () => {
  test('a tab dropped on its centre has no target; on its edge it splits', () => {
    const m = mount();
    const dt = transfer();
    dockDragStart({ tab: 'alpha' }, dt);
    fire(content(m, 't2'), 'dragover', dt, T2_CENTRE);
    expect(previewRects(m)).toEqual([]);
    fire(content(m, 't2'), 'dragover', dt, T2_BOTTOM_EDGE);
    expect(previewRects(m)).toEqual([`${PREVIEW_CLASS}__rect ${PREVIEW_CLASS}__rect--split`]);
    fire(content(m, 't2'), 'drop', dt, T2_BOTTOM_EDGE);
    // alpha's own tile emptied and left; it lands in a new tile under the Agent pane.
    expect(tabs(m.holder)).toEqual(['terminal', 'alpha']);
    expect(m.holder.layout && 'split' in m.holder.layout && m.holder.layout.split).toBe('column');
  });

  test('its own tile never tabs into another: no target on a centre, a split on an edge', () => {
    const m = mount();
    const dt = transfer();
    dockDragStart({ tile: 't2' }, dt);
    fire(content(m, 't1'), 'dragover', dt, T1_CENTRE);
    expect(previewRects(m)).toEqual([]);
    fire(content(m, 't1'), 'dragover', dt, T1_BOTTOM_EDGE);
    expect(previewRects(m)).toEqual([`${PREVIEW_CLASS}__rect ${PREVIEW_CLASS}__rect--split`]);
    fire(content(m, 't1'), 'drop', dt, T1_BOTTOM_EDGE);
    expect(shape(m.holder)).toEqual(['t1:alpha', 't2:terminal']);
  });

  test('admitted() states the rule for every source', () => {
    const agent = { tile: 't2', zone: /** @type {const} */ ('centre') };
    const service = { tile: 't1', zone: /** @type {const} */ ('centre') };
    const edge = { tile: 't2', edge: /** @type {const} */ ('left') };
    expect(admitted(AB, { tab: 'alpha' }, agent)).toBe(false);
    expect(admitted(AB, { tile: 't2' }, service)).toBe(false);
    expect(admitted(AB, { tab: 'terminal' }, service)).toBe(false);
    expect(admitted(AB, { new: 'gamma' }, service)).toBe(false);
    expect(admitted(AB, { tab: 'alpha' }, service)).toBe(true);
    expect(admitted(AB, { new: 'gamma' }, edge)).toBe(true);
    expect(admitted(AB, { tile: 't2' }, { outer: 'left' })).toBe(true);
    expect(previewKind(agent)).toBe('tab');
    expect(previewKind(edge)).toBe('split');
    expect(previewKind({ border: { tile: 't1', side: 'right' } })).toBeNull();
  });
});

describe('dockview decides nothing', () => {
  test("dockview's own tab drag is the tile; no overlay paints and nothing moves by itself", async () => {
    const m = mount();
    const seen = overlays(m);
    const before = JSON.stringify(m.api.toJSON().grid);
    const dt = transfer();
    fire(dvTab(m, 't1'), 'dragstart', dt);
    expect(dockDragSource()).toEqual({ tile: 't1' });
    expect(dragging(m)).toBe(true);
    fire(content(m, 't2'), 'dragenter', dt, T2_CENTRE);
    fire(content(m, 't2'), 'dragover', dt, T2_CENTRE);
    expect(seen.every((e) => e.defaultPrevented)).toBe(true);
    expect(m.root.querySelector('.dv-drop-target-selection')).toBeNull();
    // The Agent tile takes no tab: no target, so the drop changes nothing.
    fire(content(m, 't2'), 'drop', dt, T2_CENTRE);
    fire(dvTab(m, 't1'), 'dragend', dt);
    await settle();
    expect(JSON.stringify(m.api.toJSON().grid)).toBe(before);
    expect(shape(m.holder)).toEqual(['t1:alpha', 't2:terminal']);
    expect(dockDragSource()).toBeNull();
  });
});

describe('the gesture', () => {
  test('a drag leaving the dock leaves no highlight; Escape ends it and lowers the shield', async () => {
    const m = mount();
    const dt = transfer();
    dockDragStart({ new: 'gamma' }, dt);
    fire(content(m, 't1'), 'dragover', dt, T1_BOTTOM_EDGE);
    expect(previewRects(m)).toHaveLength(1);
    const leave = new MouseEvent('dragleave', {
      bubbles: true,
      cancelable: true,
      relatedTarget: null,
    });
    Object.defineProperty(leave, 'dataTransfer', { value: dt });
    content(m, 't1').dispatchEvent(leave);
    expect(previewRects(m)).toEqual([]);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await settle();
    expect(dragging(m)).toBe(false);
    expect(m.holder.dragLive).toBe(false);
    expect(h.shield).toEqual([true, false]);
  });

  test('teardown lowers a live drag and removes the preview', () => {
    const m = mount();
    const dt = transfer();
    dockDragStart({ new: 'gamma' }, dt);
    fire(content(m, 't1'), 'dragover', dt, T1_BOTTOM_EDGE);
    m.dispose();
    expect(previewRects(m)).toEqual([]);
    expect(dragging(m)).toBe(false);
  });
});
