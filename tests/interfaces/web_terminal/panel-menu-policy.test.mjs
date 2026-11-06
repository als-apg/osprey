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
const tree = vi.hoisted(() => ({ toggleAllSessions: vi.fn() }));

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
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js', () => tree);

const policy = await import(`${JS}/panel-menu-policy.js`);
const sidebar = await import(`${JS}/sidebar.js`);

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
    expect(policy.buildTerminalMenuItems().map((/** @type {Item} */ i) => i.label)).toEqual([
      'Reset layout',
      'Close Agent pane',
    ]);
  });

  test('the verbs reach the terminal and the dock', async () => {
    /** @type {Item[]} */
    const items = policy.buildTerminalMenuItems();
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

describe('the Agent row’s click', () => {
  /** The rail's own click on the Agent row. @param {MouseEventInit} [init] */
  const activate = (init = {}) =>
    policy
      .railOptions()
      .onActivate(TERMINAL_RAIL_ID, new MouseEvent('click', { detail: 1, ...init }));

  test('with the pane open, a click folds the session list and the next unfolds it', () => {
    sidebar.setListOpen('agent', true);
    activate();
    expect(sidebar.isListOpen('agent')).toBe(false);
    activate();
    expect(sidebar.isListOpen('agent')).toBe(true);
    expect(dock.openTerminalPanel).not.toHaveBeenCalled();
  });

  test('with the pane closed, a click opens the pane and unfolds the list', () => {
    sidebar.setListOpen('agent', false);
    dock.isTerminalPanelOpen.mockReturnValueOnce(false);
    activate();
    expect(dock.openTerminalPanel).toHaveBeenCalledTimes(1);
    expect(sidebar.isListOpen('agent')).toBe(true);
  });

  test('⌥-click toggles every session and leaves the list open and the pane alone', async () => {
    sidebar.setListOpen('agent', true);
    activate({ altKey: true });
    await vi.waitFor(() => expect(tree.toggleAllSessions).toHaveBeenCalledWith(undefined));
    expect(sidebar.isListOpen('agent')).toBe(true);
    expect(dock.openTerminalPanel).not.toHaveBeenCalled();
  });

  test('⌥-click on a folded list unfolds it onto every session open', async () => {
    sidebar.setListOpen('agent', false);
    activate({ altKey: true });
    await vi.waitFor(() => expect(tree.toggleAllSessions).toHaveBeenCalledWith(true));
    expect(sidebar.isListOpen('agent')).toBe(true);
  });

  test('Enter or Space (a click with no pointer, detail 0) stays a plain click, even with ⌥ held', async () => {
    sidebar.setListOpen('agent', true);
    activate({ altKey: true, detail: 0 });
    expect(sidebar.isListOpen('agent')).toBe(false);
    await Promise.resolve();
    expect(tree.toggleAllSessions).not.toHaveBeenCalled();
  });
});
