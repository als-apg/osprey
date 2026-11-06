/**
 * The Agent's two menus (panel-menu-policy.js): the tile header keeps the
 * pane verbs (Pop out, Export conversation in chat, Restart terminal on the
 * terminal, Reset layout, Close Agent pane), and the
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
  agentPaneContent: vi.fn(() => /** @type {HTMLElement | null} */ (null)),
}));
const chatExport = vi.hoisted(() => ({ exportConversation: vi.fn(async () => {}) }));
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
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat-export.js', () => chatExport);
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
  test('on the terminal surface: Pop out, Restart terminal, Reset layout, Close Agent pane — no New session', () => {
    const anchorEl = document.createElement('div');
    document.body.append(anchorEl);
    expect(policy.openTileContextMenu(TERMINAL_RAIL_ID, { x: 1, y: 1, anchorEl })).toBe(true);
    expect(menuLabels()).toEqual([
      'Pop out',
      'Restart terminal',
      'Reset layout',
      'Close Agent pane',
    ]);
    expect(document.querySelector('.rail-context-menu')?.getAttribute('aria-label')).toBe(
      'Agent actions'
    );
  });

  test('in chat the PTY verb is withheld and Export conversation is offered', async () => {
    surface.terminal = false;
    /** @type {Item[]} */
    const items = policy.buildTerminalMenuItems();
    expect(items.filter((i) => i.label).map((i) => i.label)).toEqual([
      'Pop out',
      'Export conversation',
      'Reset layout',
      'Close Agent pane',
    ]);
    await items.find((i) => i.label === 'Export conversation')?.run?.();
    expect(chatExport.exportConversation).toHaveBeenCalledTimes(1);
  });

  test('Pop out waits for the Agent pane, and runs the attached pop-out once it is there', () => {
    /** @type {any[]} */
    let items = policy.buildTerminalMenuItems();
    expect(items.find((i) => i.label === 'Pop out')?.disabled).toBe(true);
    dock.agentPaneContent.mockReturnValue(document.createElement('div'));
    items = policy.buildTerminalMenuItems();
    expect(items.find((i) => i.label === 'Pop out')?.disabled).toBe(false);
    dock.agentPaneContent.mockReturnValue(null);
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

  test('with every session open it offers Collapse all, the ⌥-click gesture', async () => {
    const treeEl = document.createElement('div');
    treeEl.dataset.sessionTree = '';
    const row = document.createElement('div');
    row.className = 'osprey-tree-session';
    row.dataset.sessionId = 't1';
    row.setAttribute('aria-expanded', 'true');
    treeEl.append(row);
    document.body.append(treeEl);

    policy.openRailContextMenu(TERMINAL_RAIL_ID, 1, 1);
    expect(menuLabels()).toEqual(['New session', 'Collapse all']);
    menuRow('Collapse all').click();
    await vi.waitFor(() => expect(tree.toggleAllSessions).toHaveBeenCalledWith(false));
  });

  test('Expand all expands every session through the same path as ⌥-click', async () => {
    policy.openRailContextMenu(TERMINAL_RAIL_ID, 1, 1);
    menuRow('Expand all').click();
    await vi.waitFor(() => expect(tree.toggleAllSessions).toHaveBeenCalledWith(true));
    expect(sidebar.isListOpen('agent')).toBe(true);
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
