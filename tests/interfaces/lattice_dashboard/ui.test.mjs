/**
 * Unit tests for the Lattice Dashboard UI-chrome layer (ui.js: the control
 * sidebar through the shared panel sidebar, the layout mode, the Magnets |
 * Settings switch, and the unified drag-and-drop panel rearrangement).
 *
 * happy-dom environment (configured globally) with its built-in
 * window.localStorage standing in as the mocked localStorage,
 * cleared per test for isolation -- same convention as
 * theme-settheme.test.mjs. Plotly mocked via vi.stubGlobal (the ambient
 * `Plotly` global normally comes from the vendored classic script -- see
 * vendor-globals.d.ts). panel-sidebar.js is the real module behind a spy, so
 * the call and its effect are both asserted:
 *   npx vitest run tests/interfaces/lattice_dashboard/ui.test.mjs
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { test, expect, vi, describe, afterEach, beforeEach } from 'vitest';

import { qs, byId } from '../_support/dom.mjs';

vi.mock('/design-system/js/panel-sidebar.js', async (importOriginal) => {
  const real = /** @type {typeof import('/design-system/js/panel-sidebar.js')} */ (
    await importOriginal()
  );
  return { ...real, initPanelSidebar: vi.fn(real.initPanelSidebar) };
});

import { initPanelSidebar } from '/design-system/js/panel-sidebar.js';
import { createUI } from '../../../src/osprey/interfaces/lattice_dashboard/static/js/ui.js';

const FIGURE_NAMES = ['optics', 'da'];

/** A DOM element carrying Plotly's runtime-attached `.data`/`.layout`, as set by Plotly.react(). */
/** @typedef {HTMLElement & {data?: unknown}} PlotlyPlotElement */

/** Minimal DOM fixture matching lattice_dashboard/static/index.html's structure. */
function mountFixture() {
  document.body.innerHTML = `
    <div id="panel-root" data-panel-root>
    <div class="osprey-panel-toolbar">
      <button id="btn-sidebar" class="osprey-icon-btn" aria-controls="sidebar"></button>
    </div>
    <div id="main-layout">
    <aside id="sidebar" class="osprey-panel-sidebar" data-open="false">
      <div class="osprey-seg sidebar-tabs">
        <button class="sidebar-tab" data-tab="magnets" aria-pressed="true"></button>
        <button class="sidebar-tab" data-tab="settings" aria-pressed="false"></button>
      </div>
      <div id="tab-magnets" class="sidebar-tab-content sidebar-tab-content--active"></div>
      <div id="tab-settings" class="sidebar-tab-content"></div>
    </aside>
    <div id="sidebar-resizer" class="osprey-splitter" tabindex="0"></div>
    <main id="figure-area">
      <div class="figure-grid">
        <div class="figure-cell" id="cell-optics" data-figure="optics">
          <div class="figure-header"></div>
          <div class="figure-plot" id="plot-optics"></div>
        </div>
      </div>
      <div class="verification-row">
        <div class="figure-cell figure-cell--verification" id="cell-da" data-figure="da">
          <div class="figure-header"></div>
          <div class="figure-plot" id="plot-da"></div>
        </div>
      </div>
    </main>
    </div>
    </div>
  `;
}

/** The stored sidebar record's collapsed flag. @param {string} [key] */
function storedCollapsed(key = 'lattice-sidebar-collapsed') {
  return JSON.parse(String(window.localStorage.getItem(key))).collapsed;
}

/** @param {string} tab */
const pressed = (tab) =>
  qs(document, `.sidebar-tab[data-tab="${tab}"]`).getAttribute('aria-pressed');

beforeEach(() => {
  window.localStorage.clear();
  mountFixture();
  vi.stubGlobal('Plotly', { relayout: vi.fn() });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
  vi.mocked(initPanelSidebar).mockClear();
});

describe('control sidebar', () => {
  test('is the shared panel sidebar, open by default, toggled from the toolbar', () => {
    const api = createUI(FIGURE_NAMES).initSidebar();

    expect(initPanelSidebar).toHaveBeenCalledTimes(1);
    expect(vi.mocked(initPanelSidebar).mock.calls[0][0]).toMatchObject({
      sidebar: byId('sidebar'),
      handle: byId('sidebar-resizer'),
      toggle: byId('btn-sidebar'),
      storageKey: 'lattice-sidebar-collapsed',
      defaultOpen: true,
    });
    expect(api?.isOpen()).toBe(true);
    expect(byId('sidebar').dataset.open).toBe('true');
    expect(byId('btn-sidebar').getAttribute('aria-expanded')).toBe('true');
  });

  test('initSidebar is idempotent', () => {
    const ui = createUI(FIGURE_NAMES);
    expect(ui.initSidebar()).toBe(ui.initSidebar());
    expect(initPanelSidebar).toHaveBeenCalledTimes(1);
  });

  test("the operator's toggle closes it and is stored", () => {
    createUI(FIGURE_NAMES).initSidebar();

    byId('btn-sidebar').click();
    expect(byId('sidebar').dataset.open).toBe('false');
    expect(storedCollapsed()).toBe(true);

    byId('btn-sidebar').click();
    expect(byId('sidebar').dataset.open).toBe('true');
    expect(storedCollapsed()).toBe(false);
  });

  test('a stored close carries over from the old sidebar', () => {
    window.localStorage.setItem(
      'lattice-sidebar-collapsed',
      JSON.stringify({ size: 300, collapsed: true })
    );
    createUI(FIGURE_NAMES).initSidebar();
    expect(byId('sidebar').dataset.open).toBe('false');
  });
});

describe('layout mode', () => {
  test('initLayout defaults to stacked and reports it', () => {
    const ui = createUI(FIGURE_NAMES);

    expect(ui.initLayout()).toBe('stacked');
    expect(byId('figure-area').classList.contains('layout-stacked')).toBe(true);
    expect(window.localStorage.getItem('lattice-layout-mode')).toBe('stacked');
  });

  test('initLayout honors a stored "grid" preference', () => {
    window.localStorage.setItem('lattice-layout-mode', 'grid');
    const ui = createUI(FIGURE_NAMES);

    expect(ui.initLayout()).toBe('grid');
    expect(byId('figure-area').classList.contains('layout-stacked')).toBe(false);
  });

  test('applyLayout switches stacked <-> grid and persists each mode', () => {
    const ui = createUI(FIGURE_NAMES);
    const figArea = byId('figure-area');

    ui.initLayout();
    ui.applyLayout('grid');
    expect(figArea.classList.contains('layout-stacked')).toBe(false);
    expect(ui.currentLayout()).toBe('grid');
    expect(window.localStorage.getItem('lattice-layout-mode')).toBe('grid');

    ui.applyLayout('stacked');
    expect(figArea.classList.contains('layout-stacked')).toBe(true);
    expect(window.localStorage.getItem('lattice-layout-mode')).toBe('stacked');
  });

  test('applying a layout mode reflows every figure with data via Plotly.relayout', () => {
    const ui = createUI(FIGURE_NAMES);
    // Give one figure Plotly-rendered data; the other stays a placeholder.
    /** @type {PlotlyPlotElement} */ (byId('plot-optics')).data = [{ x: [1] }];

    ui.initLayout();

    expect(Plotly.relayout).toHaveBeenCalledTimes(1);
    expect(Plotly.relayout).toHaveBeenCalledWith(byId('plot-optics'), { autosize: true });
  });
});

describe('Magnets | Settings switch', () => {
  test('a press shows that list, presses its button, and persists the choice', () => {
    const ui = createUI(FIGURE_NAMES);
    ui.initSidebarTabs();

    qs(document, '.sidebar-tab[data-tab="settings"]', HTMLElement).click();

    expect(pressed('settings')).toBe('true');
    expect(pressed('magnets')).toBe('false');
    expect(byId('tab-settings').classList.contains('sidebar-tab-content--active')).toBe(true);
    expect(byId('tab-magnets').classList.contains('sidebar-tab-content--active')).toBe(false);
    expect(window.localStorage.getItem('lattice-sidebar-tab')).toBe('settings');
  });

  test('initSidebarTabs restores the last list shown', () => {
    window.localStorage.setItem('lattice-sidebar-tab', 'settings');
    const ui = createUI(FIGURE_NAMES);
    ui.initSidebarTabs();

    expect(byId('tab-settings').classList.contains('sidebar-tab-content--active')).toBe(true);
    expect(pressed('settings')).toBe('true');
  });
});

describe('drag-and-drop panel rearrangement (unified section)', () => {
  /**
   * @typedef {{
   *   _data: Map<string, string>,
   *   effectAllowed: string,
   *   dropEffect: string,
   *   setData: (type: string, val: string) => void,
   *   getData: (type: string) => string,
   * }} FakeDataTransfer
   */

  /**
   * Simulate a full drag of `from` onto `to` via the real event sequence the
   * handlers wire. happy-dom has no native DragEvent, so a plain Event is
   * used with a hand-rolled `dataTransfer` attached — matching how ui.js's
   * handlers read it back via a DragEvent cast.
   * @param {HTMLElement} fromCell
   * @param {HTMLElement} toCell
   */
  function dragAndDrop(fromCell, toCell) {
    const fromHeader = qs(fromCell, '.figure-header');

    /** @type {FakeDataTransfer} */
    const dataTransfer = {
      _data: new Map(),
      effectAllowed: '',
      dropEffect: '',
      setData(type, val) {
        this._data.set(type, val);
      },
      getData(type) {
        return this._data.get(type) ?? '';
      },
    };

    const dragStart = new Event('dragstart', { bubbles: true });
    /** @type {Event & {dataTransfer: FakeDataTransfer}} */ (dragStart).dataTransfer = dataTransfer;
    fromHeader.dispatchEvent(dragStart);

    const drop = new Event('drop', { bubbles: true });
    /** @type {Event & {dataTransfer: FakeDataTransfer}} */ (drop).dataTransfer = dataTransfer;
    toCell.dispatchEvent(drop);
  }

  test('dropping one panel onto another swaps their DOM positions and saves the new order', () => {
    const ui = createUI(FIGURE_NAMES);
    ui.setupDragAndDrop();

    const optics = byId('cell-optics');
    const da = byId('cell-da');
    const opticsContainer = optics.parentNode;
    const daContainer = da.parentNode;

    dragAndDrop(optics, da);

    // Cells swapped containers (optics moved into the verification row, da
    // moved into the figure grid).
    expect(optics.parentNode).toBe(daContainer);
    expect(da.parentNode).toBe(opticsContainer);

    const savedRaw = window.localStorage.getItem('lattice-panel-order');
    if (savedRaw === null) throw new Error('lattice-panel-order was not saved');
    const saved = JSON.parse(savedRaw);
    expect(new Set(saved)).toEqual(new Set(['optics', 'da']));
  });

  test('restorePanelOrder replays a saved order across a fresh DOM', () => {
    // Saved order has 'da' first, but the fixture renders optics first.
    window.localStorage.setItem('lattice-panel-order', JSON.stringify(['da', 'optics']));
    const ui = createUI(FIGURE_NAMES);

    ui.restorePanelOrder();

    const opticsContainer = byId('cell-optics').parentNode;
    const firstChildOfOpticsContainer = /** @type {Element} */ (opticsContainer).querySelector(
      '.figure-cell'
    );
    // 'da' was reordered into whichever slot position 0 maps to; assert via
    // the round-trip contract instead of a hardcoded DOM shape: reading the
    // order back out via savePanelOrder's own selector must match 'saved'.
    const cellsInOrder = Array.from(document.querySelectorAll('.figure-cell')).map(
      (c) => /** @type {HTMLElement} */ (c).dataset.figure
    );
    expect(cellsInOrder).toEqual(['da', 'optics']);
    expect(firstChildOfOpticsContainer).not.toBeNull();
  });

  test('restorePanelOrder discards a stale saved order (panel renamed) instead of throwing', () => {
    window.localStorage.setItem('lattice-panel-order', JSON.stringify(['optics', 'fma'])); // 'fma' no longer exists
    const ui = createUI(FIGURE_NAMES);

    expect(() => ui.restorePanelOrder()).not.toThrow();
    expect(window.localStorage.getItem('lattice-panel-order')).toBeNull();
    // DOM order is untouched (still the fixture's natural order).
    const cellsInOrder = Array.from(document.querySelectorAll('.figure-cell')).map(
      (c) => /** @type {HTMLElement} */ (c).dataset.figure
    );
    expect(cellsInOrder).toEqual(['optics', 'da']);
  });

  test('restorePanelOrder is a no-op when nothing is saved', () => {
    const ui = createUI(FIGURE_NAMES);
    expect(() => ui.restorePanelOrder()).not.toThrow();
    const cellsInOrder = Array.from(document.querySelectorAll('.figure-cell')).map(
      (c) => /** @type {HTMLElement} */ (c).dataset.figure
    );
    expect(cellsInOrder).toEqual(['optics', 'da']);
  });

  test('panel-order save/restore round-trip: drag-and-drop save feeds back into restore on a fresh controller', () => {
    const ui1 = createUI(FIGURE_NAMES);
    ui1.setupDragAndDrop();
    dragAndDrop(byId('cell-optics'), byId('cell-da'));
    const savedOrderRaw = window.localStorage.getItem('lattice-panel-order');
    if (savedOrderRaw === null) throw new Error('lattice-panel-order was not saved');
    const savedOrder = JSON.parse(savedOrderRaw);

    // Fresh DOM + fresh controller, as a real page reload would produce.
    mountFixture();
    const ui2 = createUI(FIGURE_NAMES);
    ui2.restorePanelOrder();

    const cellsInOrder = Array.from(document.querySelectorAll('.figure-cell')).map(
      (c) => /** @type {HTMLElement} */ (c).dataset.figure
    );
    expect(cellsInOrder).toEqual(savedOrder);
  });
});

describe('on a multi-user mount (storage scope)', () => {
  const SCOPE_ATTR = 'data-osprey-storage-scope';

  /** The same drag sequence as above, reduced to what the order save needs.
   * @param {HTMLElement} fromCell @param {HTMLElement} toCell */
  function drag(fromCell, toCell) {
    /** @type {Map<string, string>} */
    const data = new Map();
    const dataTransfer = {
      effectAllowed: '',
      dropEffect: '',
      /** @param {string} t @param {string} v */
      setData(t, v) {
        data.set(t, v);
      },
      /** @param {string} t */
      getData(t) {
        return data.get(t) ?? '';
      },
    };
    for (const [type, target] of [
      ['dragstart', qs(fromCell, '.figure-header')],
      ['drop', toCell],
    ]) {
      const event = new Event(/** @type {string} */ (type), { bubbles: true });
      /** @type {Event & {dataTransfer: unknown}} */ (event).dataTransfer = dataTransfer;
      /** @type {HTMLElement} */ (target).dispatchEvent(event);
    }
  }

  const figureOrder = () =>
    Array.from(document.querySelectorAll('.figure-cell')).map(
      (c) => /** @type {HTMLElement} */ (c).dataset.figure
    );

  beforeEach(() => {
    document.documentElement.setAttribute(SCOPE_ATTR, 'bob');
  });

  afterEach(() => {
    document.documentElement.removeAttribute(SCOPE_ATTR);
  });

  test('the layout mode is read and written under the scope', () => {
    window.localStorage.setItem('lattice-layout-mode', 'grid');
    const ui = createUI(FIGURE_NAMES);
    ui.initLayout();

    expect(byId('figure-area').classList.contains('layout-stacked')).toBe(true);
    expect(window.localStorage.getItem('lattice-layout-mode--bob')).toBe('stacked');
    expect(window.localStorage.getItem('lattice-layout-mode')).toBe('grid');
  });

  test('the sidebar tab is read and written under the scope', () => {
    window.localStorage.setItem('lattice-sidebar-tab', 'settings');
    const ui = createUI(FIGURE_NAMES);
    ui.initSidebarTabs();
    expect(byId('tab-magnets').classList.contains('sidebar-tab-content--active')).toBe(true);

    qs(document, '.sidebar-tab[data-tab="settings"]', HTMLElement).click();
    expect(window.localStorage.getItem('lattice-sidebar-tab--bob')).toBe('settings');

    window.localStorage.setItem('lattice-sidebar-tab', 'magnets');
    mountFixture();
    createUI(FIGURE_NAMES).initSidebarTabs();
    expect(byId('tab-settings').classList.contains('sidebar-tab-content--active')).toBe(true);
  });

  test('the figure order is saved and restored under the scope', () => {
    window.localStorage.setItem('lattice-panel-order', JSON.stringify(['optics', 'da']));
    const ui1 = createUI(FIGURE_NAMES);
    ui1.setupDragAndDrop();
    drag(byId('cell-optics'), byId('cell-da'));
    const saved = JSON.parse(String(window.localStorage.getItem('lattice-panel-order--bob')));
    expect(saved).toEqual(['da', 'optics']);
    expect(window.localStorage.getItem('lattice-panel-order')).toBe(
      JSON.stringify(['optics', 'da'])
    );

    mountFixture();
    createUI(FIGURE_NAMES).restorePanelOrder();
    expect(figureOrder()).toEqual(['da', 'optics']);

    // A stale order is discarded from the person's own slot, never the shared one.
    window.localStorage.setItem('lattice-panel-order--bob', JSON.stringify(['optics', 'fma']));
    mountFixture();
    createUI(FIGURE_NAMES).restorePanelOrder();
    expect(window.localStorage.getItem('lattice-panel-order--bob')).toBeNull();
    expect(window.localStorage.getItem('lattice-panel-order')).not.toBeNull();
    expect(figureOrder()).toEqual(['optics', 'da']);
  });

  test('the sidebar state is persisted under the scope', () => {
    window.localStorage.setItem('lattice-sidebar-collapsed', 'true');
    createUI(FIGURE_NAMES).initSidebar();
    expect(byId('sidebar').dataset.open).toBe('true');

    byId('btn-sidebar').click();
    expect(storedCollapsed('lattice-sidebar-collapsed--bob')).toBe(true);
    expect(window.localStorage.getItem('lattice-sidebar-collapsed')).toBe('true');
  });
});

describe('the page markup', () => {
  const html = readFileSync(
    resolve(
      dirname(fileURLToPath(import.meta.url)),
      '../../../src/osprey/interfaces/lattice_dashboard/static/index.html'
    ),
    'utf8'
  );
  // A template parses the body inert: no stylesheet or script loads.
  const template = document.createElement('template');
  template.innerHTML = html.slice(html.indexOf('<body'), html.indexOf('</body>'));
  const page = template.content;

  test('the sidebar toggle is the first control of the panel toolbar', () => {
    const start = page.querySelector('.osprey-panel-toolbar__start');
    expect(start?.firstElementChild?.id).toBe('btn-sidebar');
    expect(page.getElementById('btn-sidebar')?.getAttribute('aria-controls')).toBe('sidebar');
  });

  test('the sidebar has no collapse control or footer of its own', () => {
    const sidebar = page.getElementById('sidebar');
    expect(sidebar?.classList.contains('osprey-panel-sidebar')).toBe(true);
    expect(sidebar?.querySelectorAll('button:not(.sidebar-tab)')).toHaveLength(0);
    expect(sidebar?.querySelector('.osprey-seg')?.querySelectorAll('button')).toHaveLength(2);
  });
});
