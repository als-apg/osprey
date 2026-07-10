// @ts-nocheck
// TODO(frontend-hardening): type-clean this test; tracked in eslint.config.js local/no-ts-nocheck allowlist, which may only shrink.
/**
 * Unit tests for the Lattice Dashboard UI-chrome layer (ui.js: sidebar
 * collapse, layout-mode toggle, sidebar tabs, and the unified
 * drag-and-drop panel rearrangement).
 *
 * happy-dom environment (configured globally) with its built-in
 * window.localStorage standing in as the mocked localStorage,
 * cleared per test for isolation -- same convention as
 * theme-settheme.test.mjs. Plotly mocked via vi.stubGlobal (the ambient
 * `Plotly` global normally comes from the vendored classic script -- see
 * vendor-globals.d.ts):
 *   npx vitest run tests/interfaces/lattice_dashboard/ui.test.mjs
 */

import { test, expect, vi, describe, afterEach, beforeEach } from 'vitest';

import { createUI } from '../../../src/osprey/interfaces/lattice_dashboard/static/js/ui.js';

const FIGURE_NAMES = ['optics', 'da'];

/** Minimal DOM fixture matching lattice_dashboard/static/index.html's structure. */
function mountFixture() {
  document.body.innerHTML = `
    <button id="btn-layout" class="action-btn">
      <svg class="icon-grid"></svg>
      <svg class="icon-stack" style="display:none"></svg>
      <span class="layout-label">Grid</span>
    </button>
    <aside id="sidebar" class="sidebar-collapsed">
      <div class="sidebar-tabs">
        <button class="sidebar-tab sidebar-tab--active" data-tab="magnets"></button>
        <button class="sidebar-tab" data-tab="settings"></button>
      </div>
      <button id="btn-sidebar-toggle"></button>
      <div id="tab-magnets" class="sidebar-tab-content sidebar-tab-content--active"></div>
      <div id="tab-settings" class="sidebar-tab-content"></div>
    </aside>
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
  `;
}

beforeEach(() => {
  window.localStorage.clear();
  mountFixture();
  vi.stubGlobal('Plotly', { relayout: vi.fn() });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('sidebar collapse', () => {
  test('initSidebar defaults to collapsed with no stored preference', () => {
    const ui = createUI(FIGURE_NAMES);
    document.getElementById('sidebar').classList.remove('sidebar-collapsed');
    ui.initSidebar();
    expect(document.getElementById('sidebar').classList.contains('sidebar-collapsed')).toBe(true);
  });

  test('initSidebar honors a stored "expanded" preference', () => {
    window.localStorage.setItem('lattice-sidebar-collapsed', 'false');
    const ui = createUI(FIGURE_NAMES);
    ui.initSidebar();
    expect(document.getElementById('sidebar').classList.contains('sidebar-collapsed')).toBe(false);
  });

  test('toggleSidebar flips the class and persists the new state', () => {
    const ui = createUI(FIGURE_NAMES);
    ui.toggleSidebar();
    expect(document.getElementById('sidebar').classList.contains('sidebar-collapsed')).toBe(false);
    expect(window.localStorage.getItem('lattice-sidebar-collapsed')).toBe('false');

    ui.toggleSidebar();
    expect(document.getElementById('sidebar').classList.contains('sidebar-collapsed')).toBe(true);
    expect(window.localStorage.getItem('lattice-sidebar-collapsed')).toBe('true');
  });
});

describe('layout-mode toggle', () => {
  test('initLayout defaults to stacked and applies the stacked class + button labels', () => {
    const ui = createUI(FIGURE_NAMES);
    ui.initLayout();

    const figArea = document.getElementById('figure-area');
    expect(figArea.classList.contains('layout-stacked')).toBe(true);
    expect(document.querySelector('.layout-label').textContent).toBe('Grid');
    expect(document.querySelector('.icon-grid').style.display).toBe('');
    expect(document.querySelector('.icon-stack').style.display).toBe('none');
    expect(window.localStorage.getItem('lattice-layout-mode')).toBe('stacked');
  });

  test('initLayout honors a stored "grid" preference', () => {
    window.localStorage.setItem('lattice-layout-mode', 'grid');
    const ui = createUI(FIGURE_NAMES);
    ui.initLayout();

    const figArea = document.getElementById('figure-area');
    expect(figArea.classList.contains('layout-stacked')).toBe(false);
    expect(document.querySelector('.layout-label').textContent).toBe('Stack');
  });

  test('toggleLayout flips stacked <-> grid and persists each mode', () => {
    const ui = createUI(FIGURE_NAMES);
    const figArea = document.getElementById('figure-area');

    ui.initLayout(); // stacked (default)
    ui.toggleLayout();
    expect(figArea.classList.contains('layout-stacked')).toBe(false);
    expect(window.localStorage.getItem('lattice-layout-mode')).toBe('grid');

    ui.toggleLayout();
    expect(figArea.classList.contains('layout-stacked')).toBe(true);
    expect(window.localStorage.getItem('lattice-layout-mode')).toBe('stacked');
  });

  test('applying a layout mode reflows every figure with data via Plotly.relayout', () => {
    const ui = createUI(FIGURE_NAMES);
    // Give one figure Plotly-rendered data; the other stays a placeholder.
    /** @type {any} */ (document.getElementById('plot-optics')).data = [{ x: [1] }];

    ui.initLayout();

    expect(Plotly.relayout).toHaveBeenCalledTimes(1);
    expect(Plotly.relayout).toHaveBeenCalledWith(
      document.getElementById('plot-optics'),
      { autosize: true }
    );
  });
});

describe('sidebar tabs', () => {
  test('clicking a tab switches active button and content, and persists the choice', () => {
    const ui = createUI(FIGURE_NAMES);
    ui.initSidebarTabs();

    document.querySelector('.sidebar-tab[data-tab="settings"]').click();

    expect(
      document.querySelector('.sidebar-tab[data-tab="settings"]').classList.contains('sidebar-tab--active')
    ).toBe(true);
    expect(
      document.querySelector('.sidebar-tab[data-tab="magnets"]').classList.contains('sidebar-tab--active')
    ).toBe(false);
    expect(document.getElementById('tab-settings').classList.contains('sidebar-tab-content--active')).toBe(true);
    expect(document.getElementById('tab-magnets').classList.contains('sidebar-tab-content--active')).toBe(false);
    expect(window.localStorage.getItem('lattice-sidebar-tab')).toBe('settings');
  });

  test('initSidebarTabs restores the last active tab from localStorage', () => {
    window.localStorage.setItem('lattice-sidebar-tab', 'settings');
    const ui = createUI(FIGURE_NAMES);
    ui.initSidebarTabs();

    expect(document.getElementById('tab-settings').classList.contains('sidebar-tab-content--active')).toBe(true);
  });

  test('clicking a tab while the sidebar is collapsed expands it', () => {
    const ui = createUI(FIGURE_NAMES);
    document.getElementById('sidebar').classList.add('sidebar-collapsed');
    ui.initSidebarTabs();

    document.querySelector('.sidebar-tab[data-tab="settings"]').click();

    expect(document.getElementById('sidebar').classList.contains('sidebar-collapsed')).toBe(false);
    expect(window.localStorage.getItem('lattice-sidebar-collapsed')).toBe('false');
  });
});

describe('drag-and-drop panel rearrangement (unified section)', () => {
  /** Simulate a full drag of `from` onto `to` via the real event sequence the handlers wire. */
  function dragAndDrop(fromCell, toCell) {
    const fromHeader = fromCell.querySelector('.figure-header');

    /** @type {any} */
    const dataTransfer = {
      _data: new Map(),
      effectAllowed: '',
      dropEffect: '',
      setData(type, val) { this._data.set(type, val); },
      getData(type) { return this._data.get(type) ?? ''; },
    };

    const dragStart = new Event('dragstart', { bubbles: true });
    dragStart.dataTransfer = dataTransfer;
    fromHeader.dispatchEvent(dragStart);

    const drop = new Event('drop', { bubbles: true });
    drop.dataTransfer = dataTransfer;
    toCell.dispatchEvent(drop);
  }

  test('dropping one panel onto another swaps their DOM positions and saves the new order', () => {
    const ui = createUI(FIGURE_NAMES);
    ui.setupDragAndDrop();

    const optics = document.getElementById('cell-optics');
    const da = document.getElementById('cell-da');
    const opticsContainer = optics.parentNode;
    const daContainer = da.parentNode;

    dragAndDrop(optics, da);

    // Cells swapped containers (optics moved into the verification row, da
    // moved into the figure grid).
    expect(optics.parentNode).toBe(daContainer);
    expect(da.parentNode).toBe(opticsContainer);

    const saved = JSON.parse(window.localStorage.getItem('lattice-panel-order'));
    expect(new Set(saved)).toEqual(new Set(['optics', 'da']));
  });

  test('restorePanelOrder replays a saved order across a fresh DOM', () => {
    // Saved order has 'da' first, but the fixture renders optics first.
    window.localStorage.setItem('lattice-panel-order', JSON.stringify(['da', 'optics']));
    const ui = createUI(FIGURE_NAMES);

    ui.restorePanelOrder();

    const opticsContainer = document.getElementById('cell-optics').parentNode;
    const firstChildOfOpticsContainer = /** @type {Element} */ (opticsContainer).querySelector('.figure-cell');
    // 'da' was reordered into whichever slot position 0 maps to; assert via
    // the round-trip contract instead of a hardcoded DOM shape: reading the
    // order back out via savePanelOrder's own selector must match 'saved'.
    const cellsInOrder = Array.from(document.querySelectorAll('.figure-cell')).map(c => c.dataset.figure);
    expect(cellsInOrder).toEqual(['da', 'optics']);
    expect(firstChildOfOpticsContainer).not.toBeNull();
  });

  test('restorePanelOrder discards a stale saved order (panel renamed) instead of throwing', () => {
    window.localStorage.setItem('lattice-panel-order', JSON.stringify(['optics', 'fma'])); // 'fma' no longer exists
    const ui = createUI(FIGURE_NAMES);

    expect(() => ui.restorePanelOrder()).not.toThrow();
    expect(window.localStorage.getItem('lattice-panel-order')).toBeNull();
    // DOM order is untouched (still the fixture's natural order).
    const cellsInOrder = Array.from(document.querySelectorAll('.figure-cell')).map(c => c.dataset.figure);
    expect(cellsInOrder).toEqual(['optics', 'da']);
  });

  test('restorePanelOrder is a no-op when nothing is saved', () => {
    const ui = createUI(FIGURE_NAMES);
    expect(() => ui.restorePanelOrder()).not.toThrow();
    const cellsInOrder = Array.from(document.querySelectorAll('.figure-cell')).map(c => c.dataset.figure);
    expect(cellsInOrder).toEqual(['optics', 'da']);
  });

  test('panel-order save/restore round-trip: drag-and-drop save feeds back into restore on a fresh controller', () => {
    const ui1 = createUI(FIGURE_NAMES);
    ui1.setupDragAndDrop();
    dragAndDrop(document.getElementById('cell-optics'), document.getElementById('cell-da'));
    const savedOrder = JSON.parse(window.localStorage.getItem('lattice-panel-order'));

    // Fresh DOM + fresh controller, as a real page reload would produce.
    mountFixture();
    const ui2 = createUI(FIGURE_NAMES);
    ui2.restorePanelOrder();

    const cellsInOrder = Array.from(document.querySelectorAll('.figure-cell')).map(c => c.dataset.figure);
    expect(cellsInOrder).toEqual(savedOrder);
  });
});
