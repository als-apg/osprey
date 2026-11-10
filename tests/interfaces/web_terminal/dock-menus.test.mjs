// @ts-check
/**
 * The empty dock slot's place menu (dock-menus.js):
 *   npx vitest run tests/interfaces/web_terminal/dock-menus.test.mjs
 *
 * Covered: the rows and their order (Open panel ▸ · Reset layout, destructive,
 * last, behind a divider); Open panel lists the rail's service panels in
 * sidebar order, leaving out the Agent row, parked rows, panels in a tile and
 * detached panels; never-started, failed (with the server's message) and
 * unreachable rows, and panels the manager reports unhealthy, are listed
 * disabled with the reason; one disabled row when nothing is left; a row runs
 * the injected Open verb, Reset layout the injected reset. The place: #dock-root
 * matches only while the dock holds no tile (right-click on it or its
 * watermark, and the menu key on the focused, tabindex=-1 root); a tile
 * placeholder body, and #dock-root itself, do not match while tiles exist; the
 * fallback "no active panel" card matches only in fallback (no holder), and a
 * click on the slot focuses it. untiledPanels() in fallback leaves out the
 * surfaced panel.
 *
 * context-menu.js and place-menu.js are the real modules; dock-apply.js is
 * mocked so each test sets the holder (null = fallback) and the detached set.
 * Module state is isolated by vi.resetModules() + dynamic import per test.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/dock-menus.js';
const PLACES = '../../../src/osprey/interfaces/web_terminal/static/js/place-menu.js';
const MENU = '../../../src/osprey/interfaces/design_system/static/js/context-menu.js';

/** The dock state the mocked dock-apply.js reports. */
const dock = vi.hoisted(() => ({
  /** @type {{ layout: any } | null} */
  holder: null,
  /** @type {Set<string>} */
  detached: new Set(),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js', () => ({
  getLayoutHolder: () => dock.holder,
  isDetached: (/** @type {string} */ id) => dock.detached.has(id),
}));

/** A layout with ARIEL and the Agent in tiles. */
const LAYOUT = {
  split: 'row',
  items: [
    { tile: 't1', tabs: ['ariel'], showing: 'ariel', size: 0.6 },
    { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
  ],
};

/** @type {any} */ let dockMenus;
/** @type {any} */ let pm;
/** @type {any} */ let cm;
/** @type {(() => void) | null} */ let uninstall = null;
/** @type {{ activate: any, isHealthy: any, labelOf: any, reset: any }} */ let spies;

/**
 * A rail row.
 * @param {string} id
 * @param {string} label
 * @param {string} [cls]
 * @param {string} [extra]
 */
function railRow(id, label, cls = '', extra = '') {
  return `<button class="osprey-nav-row panel-rail-button ${cls}" data-panel-id="${id}" ${extra}>
    <span class="osprey-nav-label">${label}</span></button>`;
}

const RAIL = `
  <nav id="panel-rail">
    ${railRow('terminal', 'Agent')}
    ${railRow('ariel', 'ARIEL')}
    ${railRow('channels', 'Channels')}
    ${railRow('notebook', 'Notebook', 'disabled')}
    ${railRow('archiver', 'Archiver', 'failed', 'data-title-status="Port 8091 is in use."')}
    ${railRow('logbook', 'Logbook', 'unreachable')}
    ${railRow('tuning', 'Tuning')}
    <div class="sidebar-row" hidden>${railRow('parked', 'Parked')}</div>
  </nav>`;

beforeEach(async () => {
  vi.resetModules();
  dock.holder = { layout: null };
  dock.detached = new Set();
  cm = await import(MENU);
  pm = await import(PLACES);
  dockMenus = await import(MODULE);
  document.body.innerHTML = `
    ${RAIL}
    <div id="dock-root" class="dock-root">
      <div class="dv-watermark-container"><div class="dv-watermark"></div></div>
    </div>
    <div id="right-panel"><div id="panel-content"></div></div>
    <div class="elsewhere"></div>`;
  spies = {
    activate: vi.fn(),
    isHealthy: vi.fn(() => true),
    labelOf: vi.fn((/** @type {string} */ id) => id),
    reset: vi.fn(),
  };
  uninstall = dockMenus.installDockMenus(spies);
  pm.installPlaceMenus(document);
});

afterEach(() => {
  uninstall?.();
  uninstall = null;
  cm.closeContextMenu();
  pm.resetPlaceMenus([document]);
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

/** @param {string} sel @returns {HTMLElement} */
function el(sel) {
  const node = document.querySelector(sel);
  if (!node) throw new Error(`no ${sel}`);
  return /** @type {HTMLElement} */ (node);
}

/**
 * Right-click `target`.
 * @param {Element} target
 */
function rightClick(target) {
  const e = new MouseEvent('contextmenu', {
    bubbles: true,
    cancelable: true,
    button: 2,
    clientX: 40,
    clientY: 50,
  });
  target.dispatchEvent(e);
  return e;
}

function menus() {
  return document.querySelectorAll('.osprey-context-menu[role="menu"]');
}

function rowLabels() {
  return [...document.querySelectorAll('.osprey-context-menu [role="menuitem"]')].map(
    (b) => b.querySelector('.osprey-context-label')?.textContent
  );
}

/** Put the fallback card into #panel-content, as renderEmptyState does. */
function renderFallbackCard() {
  el('#panel-content').innerHTML = `
    <div class="artifacts-empty-state">
      <div class="artifacts-empty-title">Services</div>
      <div class="artifacts-empty-text">No active panel</div>
    </div>`;
}

/** The Open panel submenu's rows. */
function openPanelRows() {
  const items = dockMenus.buildEmptyDockItems();
  return items[0].submenu;
}

describe('rows', () => {
  test('Open panel ▸ then Reset layout, destructive and last behind a divider', () => {
    const items = dockMenus.buildEmptyDockItems();
    expect(items).toHaveLength(3);
    expect(items[0].label).toBe('Open panel');
    expect(Array.isArray(items[0].submenu)).toBe(true);
    expect(items[1]).toEqual({ divider: true });
    expect(items[2].label).toBe('Reset layout');
    expect(items[2].danger).toBe(true);
    expect(items.filter((/** @type {any} */ i) => i.danger)).toHaveLength(1);
  });

  test('Open panel lists service panels in sidebar order, without the Agent or parked rows', () => {
    expect(openPanelRows().map((/** @type {any} */ r) => r.label)).toEqual([
      'ARIEL',
      'Channels',
      'Notebook',
      'Archiver',
      'Logbook',
      'Tuning',
    ]);
  });

  test('panels in a tile and detached panels are left out', () => {
    dock.holder = { layout: LAYOUT };
    dock.detached = new Set(['channels']);
    const ids = dockMenus.untiledPanels().map((/** @type {any} */ p) => p.id);
    expect(ids).not.toContain('ariel');
    expect(ids).not.toContain('channels');
    expect(ids).not.toContain('terminal');
    expect(ids).toEqual(['notebook', 'archiver', 'logbook', 'tuning']);
  });

  test('unhealthy and never-started panels are disabled with the reason', () => {
    spies.isHealthy.mockImplementation((/** @type {string} */ id) => id !== 'tuning');
    const byLabel = Object.fromEntries(openPanelRows().map((/** @type {any} */ r) => [r.label, r]));
    expect(byLabel.ARIEL.disabled).toBeFalsy();
    expect(byLabel.Channels.disabled).toBeFalsy();
    expect(byLabel.Notebook).toMatchObject({ disabled: true, tip: dockMenus.NOT_STARTED_TIP });
    expect(byLabel.Archiver).toMatchObject({ disabled: true, tip: 'Port 8091 is in use.' });
    expect(byLabel.Logbook).toMatchObject({ disabled: true, tip: dockMenus.UNREACHABLE_TIP });
    expect(byLabel.Tuning).toMatchObject({ disabled: true, tip: dockMenus.UNHEALTHY_TIP });
  });

  test('one disabled row when nothing is left to open', () => {
    el('#panel-rail').innerHTML = railRow('terminal', 'Agent') + railRow('ariel', 'ARIEL');
    dock.detached = new Set(['ariel']);
    expect(openPanelRows()).toEqual([{ label: dockMenus.NO_PANELS_LABEL, disabled: true }]);
  });

  test('a panel row runs the injected Open verb; Reset layout the injected reset', () => {
    const ariel = openPanelRows().find((/** @type {any} */ r) => r.label === 'ARIEL');
    ariel.run();
    expect(spies.activate).toHaveBeenCalledWith('ariel');
    dockMenus.buildEmptyDockItems()[2].run();
    expect(spies.reset).toHaveBeenCalledTimes(1);
  });

  test('in fallback, the surfaced panel counts as tiled', () => {
    dock.holder = null;
    el('#right-panel').setAttribute('data-active-panel', 'channels');
    const ids = dockMenus.untiledPanels().map((/** @type {any} */ p) => p.id);
    expect(ids).toContain('ariel');
    expect(ids).not.toContain('channels');
  });
});

describe('place', () => {
  test('#dock-root is focusable', () => {
    expect(el('#dock-root').getAttribute('tabindex')).toBe('-1');
  });

  test('right-click on the empty dock (its watermark) opens the menu', () => {
    const e = rightClick(el('.dv-watermark'));
    expect(e.defaultPrevented).toBe(true);
    expect(menus()).toHaveLength(1);
    expect(rowLabels()).toEqual(['Open panel', 'Reset layout']);
  });

  test('the menu key on the focused empty dock opens the menu', () => {
    const root = el('#dock-root');
    root.focus();
    const e = new KeyboardEvent('keydown', {
      key: 'F10',
      shiftKey: true,
      bubbles: true,
      cancelable: true,
    });
    root.dispatchEvent(e);
    expect(e.defaultPrevented).toBe(true);
    expect(menus()).toHaveLength(1);
    expect(rowLabels()).toEqual(['Open panel', 'Reset layout']);
  });

  test('a tile placeholder body does not match while tiles exist', () => {
    dock.holder = { layout: LAYOUT };
    el('#dock-root').innerHTML = `
      <div class="dv-groupview"><div class="dv-content-container">
        <div class="tile-placeholder-body"></div>
      </div></div>
      <div class="dv-sash"></div>`;
    for (const sel of ['.tile-placeholder-body', '.dv-sash', '#dock-root']) {
      const e = rightClick(el(sel));
      expect(e.defaultPrevented).toBe(false);
      expect(menus()).toHaveLength(0);
      expect(dockMenus.isEmptyDockSlot(el(sel))).toBe(false);
    }
  });

  test('elsewhere does not match', () => {
    const e = rightClick(el('.elsewhere'));
    expect(e.defaultPrevented).toBe(false);
    expect(menus()).toHaveLength(0);
  });

  test('the fallback card matches only in fallback', () => {
    renderFallbackCard();
    const docked = rightClick(el('.artifacts-empty-text'));
    expect(docked.defaultPrevented).toBe(false);
    expect(menus()).toHaveLength(0);

    dock.holder = null;
    const fallback = rightClick(el('.artifacts-empty-text'));
    expect(fallback.defaultPrevented).toBe(true);
    expect(rowLabels()).toEqual(['Open panel', 'Reset layout']);
  });

  test('in fallback, #dock-root is not the slot', () => {
    dock.holder = null;
    expect(dockMenus.isEmptyDockSlot(el('#dock-root'))).toBe(false);
  });

  test('a click on the fallback card gives it tabindex and focuses it', () => {
    dock.holder = null;
    renderFallbackCard();
    el('.artifacts-empty-title').click();
    const card = el('.artifacts-empty-state');
    expect(card.getAttribute('tabindex')).toBe('-1');
    expect(document.activeElement).toBe(card);
  });

  test('uninstall removes the place', () => {
    uninstall?.();
    uninstall = null;
    const e = rightClick(el('.dv-watermark'));
    expect(e.defaultPrevented).toBe(false);
    expect(menus()).toHaveLength(0);
  });
});
