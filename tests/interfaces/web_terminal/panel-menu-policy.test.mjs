/**
 * The Agent's two menus (panel-menu-policy.js): the tile header keeps the
 * pane verbs (Restart terminal, Reset layout, Close Agent pane), and the
 * Agent row in the sidebar — the head of the session tree — gets the tree's
 * menu (New session, Expand/Collapse all), New session through switchTo.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-menu-policy.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const surface = vi.hoisted(() => ({ terminal: true }));
const dock = vi.hoisted(() => ({
  openTerminalPanel: vi.fn(),
  closeTerminalPanel: vi.fn(),
  isTerminalPanelOpen: vi.fn(() => true),
  resetDockLayout: vi.fn(),
}));
const term = vi.hoisted(() => ({
  focusTerminal: vi.fn(),
  restartTerminal: vi.fn(async () => {}),
  startTerminal: vi.fn(),
}));
const sessionModel = vi.hoisted(() => ({
  switchTo: vi.fn(async () => true),
  rows: vi.fn(() => []),
  refresh: vi.fn(),
}));
const sessions = vi.hoisted(() => ({ startNewSession: vi.fn() }));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/agent-surface.js', () => ({
  isTerminalSurface: () => surface.terminal,
  onSurfaceChange: () => {},
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => dock);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => term);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/session-model.js',
  () => sessionModel
);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sessions.js', () => sessions);

const policy = await import(`${JS}/panel-menu-policy.js`);

/** @typedef {{label?: string, run?: () => unknown}} Item */
const { TERMINAL_RAIL_ID } = await import(`${JS}/panel-catalog.js`);

/** @returns {string[]} the open menu's row labels */
function menuLabels() {
  return [...document.querySelectorAll('.rail-context-item .rail-context-label')].map(
    (n) => n.textContent ?? ''
  );
}

/** @param {string} label */
function menuRow(label) {
  return /** @type {HTMLElement} */ (
    [...document.querySelectorAll('.rail-context-item')].find(
      (r) => r.querySelector('.rail-context-label')?.textContent === label
    )
  );
}

/** @type {HTMLElement} */
let rail;

beforeEach(() => {
  document.body.innerHTML = '';
  surface.terminal = true;
  vi.clearAllMocks();
  rail = document.createElement('ul');
  const row = document.createElement('li');
  row.className = 'rail-entry';
  row.dataset.id = TERMINAL_RAIL_ID;
  const btn = document.createElement('button');
  btn.className = 'panel-rail-button';
  btn.dataset.panelId = TERMINAL_RAIL_ID;
  row.append(btn);
  rail.append(row);
  document.body.append(rail);
  policy.initMenuPolicy({
    getRailEl: () => rail,
    isMember: () => true,
    getActiveTabId: () => null,
    activateTab: () => {},
    showPanel: () => {},
    retireTile: () => {},
    labelOf: (/** @type {string} */ id) => id,
    retryStart: () => {},
  });
});

afterEach(() => {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  document.body.innerHTML = '';
});

describe('the Agent tile header', () => {
  test('on the terminal surface: Restart terminal, Reset layout, Close Agent pane — no New session', () => {
    const anchorEl = document.createElement('div');
    document.body.append(anchorEl);
    expect(policy.openTileContextMenu(TERMINAL_RAIL_ID, { x: 1, y: 1, anchorEl })).toBe(true);
    expect(menuLabels()).toEqual(['Restart terminal', 'Reset layout', 'Close Agent pane']);
    expect(document.querySelector('.rail-context-menu')?.getAttribute('aria-label')).toBe(
      'Agent actions'
    );
  });

  test('in chat the PTY verb is withheld', () => {
    surface.terminal = false;
    expect(policy.buildTerminalTileMenuItems().map((/** @type {Item} */ i) => i.label)).toEqual([
      'Reset layout',
      'Close Agent pane',
    ]);
  });

  test('the verbs reach the terminal and the dock', async () => {
    /** @type {Item[]} */
    const items = policy.buildTerminalTileMenuItems();
    await items.find((i) => i.label === 'Restart terminal')?.run?.();
    expect(term.restartTerminal).toHaveBeenCalled();
    expect(term.startTerminal).toHaveBeenCalled();
    items.find((i) => i.label === 'Reset layout')?.run?.();
    expect(dock.resetDockLayout).toHaveBeenCalledTimes(1);
  });
});

describe('the Agent row', () => {
  test('offers the session tree’s menu, not the pane verbs', () => {
    expect(policy.openRailContextMenu(TERMINAL_RAIL_ID, 1, 1)).toBe(true);
    expect(menuLabels()).toEqual(['New session', 'Expand all']);
    expect(document.querySelector('.rail-context-menu')?.getAttribute('aria-label')).toBe(
      'Agent actions'
    );
  });

  test('with a session open it offers Collapse all, which closes every open row', () => {
    const tree = document.createElement('div');
    tree.dataset.sessionTree = '';
    const row = document.createElement('div');
    row.className = 'osprey-tree-session';
    row.dataset.sessionId = 't1';
    row.setAttribute('aria-expanded', 'true');
    const disc = document.createElement('span');
    disc.dataset.treeDisc = '';
    // The controller's toggle, stood in for: a disclosure click flips the row.
    disc.addEventListener('click', () =>
      row.setAttribute('aria-expanded', String(row.getAttribute('aria-expanded') !== 'true'))
    );
    row.append(disc);
    tree.append(row);
    document.body.append(tree);

    policy.openRailContextMenu(TERMINAL_RAIL_ID, 1, 1);
    expect(menuLabels()).toEqual(['New session', 'Collapse all']);
    menuRow('Collapse all').click();
    expect(row.getAttribute('aria-expanded')).toBe('false');
  });

  test('New session goes through switchTo({new: true}), never startNewSession directly', async () => {
    policy.openRailContextMenu(TERMINAL_RAIL_ID, 1, 1);
    menuRow('New session').click();
    await vi.waitFor(() => expect(sessionModel.switchTo).toHaveBeenCalledWith({ new: true }));
    expect(sessions.startNewSession).not.toHaveBeenCalled();
  });
});
