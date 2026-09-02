// @ts-check
/**
 * A hidden bar comes back from any panel surface's context menu:
 *   npx vitest run tests/interfaces/web_terminal/js/panel-menu-restore-bars.test.js
 *
 * The header and the status bar are hidden from their OWN right-click menu,
 * and a hidden bar has no surface left to right-click. The surfaces that
 * remain — a tile header, a rail entry — must therefore offer the way back.
 * panel-menu-policy.js appends one "Show <bar>" row per hidden bar below the
 * panel's own verbs, bound to the same setBarVisible the bar's menu and the
 * Customize sheet call. These tests pin the rows: absent while both bars
 * show, present per hidden bar, effective (a PUT that turns the flag back
 * on), and offered even where the surface's own verbs are empty — the
 * simple-mode terminal, which otherwise declines the menu altogether.
 *
 * The panel modules the policy's verbs reach into (the dock, the PTY, the
 * rail) are stubbed: nothing here runs a verb, and importing dockview into
 * happy-dom is not what this suite is for.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';
import { boot, doc, endpoint, putBodies, settle, teardown } from './bar-customize-fixture.mjs';

const JS = '../../../../src/osprey/interfaces/web_terminal/static/js/';
const POLICY_PATH = `${JS}panel-menu-policy.js`;

vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  restartTerminal: vi.fn(),
  startTerminal: vi.fn(),
}));
vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/sessions.js', () => ({ startNewSession: vi.fn() }));
vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  openTerminalPanel: vi.fn(),
  closeTerminalPanel: vi.fn(),
}));
vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/panel-placement.js', () => ({ openPanelBeside: vi.fn() }));
vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/panel-commands.js', () => ({ setPanelVisibility: vi.fn() }));
vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/rail-drag.js', () => ({ railDragStart: vi.fn(), railDragEnd: vi.fn() }));
vi.mock('../../../../src/osprey/interfaces/web_terminal/static/js/panel-rail.js', () => ({ getEntry: () => null }));

/** @type {any} */
let modules;
/** @type {any} */
let policy;

/**
 * Boot the bars on a layout, then the menu policy on the same module graph.
 * @param {Record<string, unknown>} layout
 * @param {string} [uiMode]
 */
async function bootWith(layout, uiMode = 'expert') {
  modules = await boot({ fetch: endpoint({ get: layout }), uiMode });
  policy = await import(POLICY_PATH);
  policy.initMenuPolicy({
    getRailEl: () => document.createElement('nav'),
    isMember: () => true,
    getActiveTabId: () => null,
    activateTab: vi.fn(),
    showPanel: vi.fn(),
    retireTile: vi.fn(),
    labelOf: () => 'ARIEL',
    getPanelStandaloneUrl: () => null,
    popoutPanel: vi.fn(),
  });
}

/** Open a tile header's menu for `id`; returns whether one opened. @param {string} id */
function openTileMenu(id) {
  const anchor = document.createElement('div');
  document.body.append(anchor);
  return policy.openTileContextMenu(id, { x: 10, y: 10, anchorEl: anchor });
}

/** The open menu's row labels, in order. */
function labels() {
  return [...document.querySelectorAll('.rail-context-menu .rail-context-label')].map(
    (el) => el.textContent
  );
}

/** Click the row with this label. @param {string} text */
function pick(text) {
  const row = [...document.querySelectorAll('.rail-context-item')].find(
    (el) => el.querySelector('.rail-context-label')?.textContent === text
  );
  if (!row) throw new Error(`no row ${text}`);
  /** @type {HTMLElement} */ (row).click();
}

afterEach(() => {
  document.querySelector('.rail-context-menu')?.remove();
  teardown(modules);
});

describe('the Show rows on a panel surface', () => {
  test('with both bars showing a tile header offers only its own verbs', async () => {
    await bootWith(doc(['logo', 'clock'], ['stopwatch']));

    expect(openTileMenu('terminal')).toBe(true);
    expect(labels()).toEqual(['Restart terminal', 'New session', 'Close terminal tile']);
  });

  test('a hidden header adds Show header below the verbs, and it brings the header back', async () => {
    await bootWith(doc(['logo', 'clock'], ['stopwatch'], { headerVisible: false }));

    expect(openTileMenu('iframe:ariel')).toBe(true);
    expect(labels()).toEqual([
      'Focus ARIEL',
      'Open in a new tile',
      'Open in a new window',
      'Remove from rail',
      'Show header',
    ]);
    expect(document.querySelectorAll('.rail-context-divider')).toHaveLength(2);

    pick('Show header');
    await settle();

    expect(document.querySelector('.rail-context-menu')).toBe(null);
    expect(putBodies()).toHaveLength(1);
    expect(putBodies()[0].header_visible).toBe(true);
    expect(putBodies()[0].status_visible).toBe(true);
    expect(modules.customize.barVisible('header')).toBe(true);
  });

  test('each hidden bar gets its own row, and the rows are gone once both show', async () => {
    await bootWith(doc(['logo'], ['stopwatch'], { headerVisible: false, statusVisible: false }));

    openTileMenu('terminal');
    expect(labels().slice(-2)).toEqual(['Show header', 'Show status bar']);

    pick('Show status bar');
    await settle();
    expect(putBodies().at(-1).status_visible).toBe(true);
    expect(putBodies().at(-1).header_visible).toBe(false);

    openTileMenu('terminal');
    expect(labels().slice(-1)).toEqual(['Show header']);
    pick('Show header');
    await settle();

    openTileMenu('terminal');
    expect(labels()).toEqual(['Restart terminal', 'New session', 'Close terminal tile']);
  });

  test('the simple-mode terminal declines with both bars showing, but opens for a hidden one', async () => {
    await bootWith(doc(['logo'], ['stopwatch']), 'simple');
    expect(openTileMenu('terminal')).toBe(false);
    expect(document.querySelector('.rail-context-menu')).toBe(null);
    teardown(modules);

    await bootWith(doc(['logo'], ['stopwatch'], { headerVisible: false }), 'simple');
    expect(openTileMenu('terminal')).toBe(true);
    expect(labels()).toEqual(['Show header']);
    expect(document.querySelectorAll('.rail-context-divider')).toHaveLength(0);
  });
});
