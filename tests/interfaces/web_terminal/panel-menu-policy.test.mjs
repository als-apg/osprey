/**
 * The Agent's two menus (panel-menu-policy.js): the tile header keeps the
 * pane verbs (Open in new window, Export conversation in chat, Restart terminal on the
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
}));
const paneWindow = vi.hoisted(() => ({
  detach: vi.fn(),
  detachBlocker: vi.fn(() => /** @type {string | null} */ (null)),
  focusWindow: vi.fn(),
  reattach: vi.fn(),
  isDetached: vi.fn(() => false),
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
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/pane-window.js', () => paneWindow);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js',
  async (original) => ({
    .../** @type {object} */ (await original()),
    isDetached: paneWindow.isDetached,
  })
);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => term);
const frames = vi.hoisted(() => ({ reloadPanel: vi.fn(() => true) }));
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js',
  async (original) => ({ .../** @type {object} */ (await original()), ...frames })
);
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
  test('on the terminal surface: Open in new window, Restart terminal, Reset layout, Close Agent pane — no New session', () => {
    const anchorEl = document.createElement('div');
    document.body.append(anchorEl);
    expect(policy.openTileContextMenu(TERMINAL_RAIL_ID, { x: 1, y: 1, anchorEl })).toBe(true);
    expect(menuLabels()).toEqual([
      'Open in new window',
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
      'Open in new window',
      'Export conversation',
      'Reset layout',
      'Close Agent pane',
    ]);
    await items.find((i) => i.label === 'Export conversation')?.run?.();
    expect(chatExport.exportConversation).toHaveBeenCalledTimes(1);
  });

  test('the Sessions row with the pane in its window: Reattach and Close Agent pane', () => {
    paneWindow.isDetached.mockReturnValue(true);
    policy.openRailContextMenu(TERMINAL_RAIL_ID, 0, 0);
    expect(menuLabels()).toContain('Reattach');
    expect(menuLabels()).toContain('Close Agent pane');
    expect(menuLabels()).not.toContain('Open in new window');
    expect(menuLabels()).toContain('New session');
    menuRow('Reattach').click();
    expect(paneWindow.reattach).toHaveBeenCalledWith(TERMINAL_RAIL_ID, undefined);
    paneWindow.isDetached.mockReturnValue(false);
  });

  test('a click on the Sessions row brings its window forward; on the icon it reattaches', () => {
    paneWindow.isDetached.mockReturnValue(true);
    const row = rail.querySelector('button');
    const trail = document.createElement('span');
    trail.setAttribute('data-detached-trail', '');
    row?.append(trail);
    const { onActivate } = policy.railOptions();

    onActivate(TERMINAL_RAIL_ID, { target: row, detail: 1, altKey: false });
    expect(paneWindow.focusWindow).toHaveBeenCalledExactlyOnceWith(TERMINAL_RAIL_ID);
    expect(paneWindow.reattach).not.toHaveBeenCalled();

    onActivate(TERMINAL_RAIL_ID, { target: trail, detail: 1, altKey: false });
    expect(paneWindow.reattach).toHaveBeenCalledExactlyOnceWith(TERMINAL_RAIL_ID);
    expect(paneWindow.focusWindow).toHaveBeenCalledTimes(1);
    paneWindow.isDetached.mockReturnValue(false);
  });

  test('Open in new window follows the window seam: disabled with its reason, Reattach while detached', () => {
    paneWindow.detachBlocker.mockReturnValue('The Agent pane is not ready.');
    /** @type {any[]} */
    let items = policy.buildTerminalMenuItems();
    expect(items.find((i) => i.label === 'Open in new window')?.disabled).toBe(true);
    paneWindow.detachBlocker.mockReturnValue(null);
    items = policy.buildTerminalMenuItems();
    const row = items.find((i) => i.label === 'Open in new window');
    expect(row?.disabled).toBe(false);
    row?.run?.();
    expect(paneWindow.detach).toHaveBeenCalledExactlyOnceWith(TERMINAL_RAIL_ID);
    paneWindow.isDetached.mockReturnValue(true);
    items = policy.buildTerminalMenuItems();
    items.find((i) => i.label === 'Reattach')?.run?.();
    expect(paneWindow.reattach).toHaveBeenCalledExactlyOnceWith(TERMINAL_RAIL_ID, undefined);
    paneWindow.isDetached.mockReturnValue(false);
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

describe('the Sessions row', () => {
  test('with the pane closed it also opens the pane in a window', () => {
    dock.isTerminalPanelOpen.mockReturnValue(false);
    expect(policy.openRailContextMenu(TERMINAL_RAIL_ID, 1, 1)).toBe(true);
    expect(menuLabels()).toContain('Open in new window');
    /** @type {HTMLElement | undefined} */ (
      [...document.querySelectorAll('.rail-context-item')].find((n) =>
        n.textContent?.includes('Open in new window')
      )
    )?.click();
    expect(paneWindow.detach).toHaveBeenCalledWith(TERMINAL_RAIL_ID);
    dock.isTerminalPanelOpen.mockReturnValue(true);
    expect(policy.openRailContextMenu(TERMINAL_RAIL_ID, 1, 1)).toBe(true);
    expect(menuLabels()).not.toContain('Open in new window');
    document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
  });

  test('offers the session tree’s menu, not the pane verbs', () => {
    expect(policy.openRailContextMenu(TERMINAL_RAIL_ID, 1, 1)).toBe(true);
    expect(menuLabels()).toEqual(['New session', 'Expand all', 'Show pinned sessions']);
    expect(document.querySelector('.rail-context-menu')?.getAttribute('aria-label')).toBe(
      'Sessions actions'
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
    expect(menuLabels()).toEqual(['New session', 'Collapse all', 'Show pinned sessions']);
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

describe('the Agent bar in its own window', () => {
  const labels = (/** @type {any[]} */ items) => items.map((i) => i.label).filter(Boolean);

  test('offers nothing that acts on the dock', () => {
    surface.terminal = true;
    expect(labels(policy.buildTerminalMenuItems({ mode: 'detached' }))).toEqual([
      'Open in new window',
      'Restart terminal',
    ]);
    surface.terminal = false;
    expect(labels(policy.buildTerminalMenuItems({ mode: 'detached' }))).toEqual([
      'Open in new window',
      'Export conversation',
    ]);
  });

  test('the docked bar is as it was', () => {
    surface.terminal = false;
    expect(labels(policy.buildTerminalMenuItems())).toContain('Reset layout');
    expect(labels(policy.buildTerminalMenuItems({ mode: 'tile' }))).toContain('Close Agent pane');
  });
});

describe('a service tile ⋯', () => {
  /** @param {any[]} items */
  const rows = (items) =>
    items.map((i) => i.heading ?? i.label).filter((l) => l && !String(l).startsWith('Move to'));

  afterEach(() => {
    document.documentElement.removeAttribute('data-docs-url');
    policy.setServiceMenuExtras('jupyter', null);
    paneWindow.isDetached.mockReturnValue(false);
  });

  test('docked: the Tile verbs alone, with no heading, and Reload reaches the frame', () => {
    const items = policy.buildTileMenuItems('ariel');
    expect(rows(items)).toEqual([
      'Open in a new tile',
      'Open in new window',
      'Reload',
      'Remove from rail',
      'Reset layout',
    ]);
    items.find((/** @type {any} */ i) => i.label === 'Reload').run();
    expect(frames.reloadPanel).toHaveBeenCalledExactlyOnceWith('ariel');
  });

  test("the panel's own rows head the menu as This panel, and Help shows with a docs page", () => {
    document.documentElement.dataset.docsUrl = 'https://docs.example/';
    policy.setServiceMenuExtras('jupyter', () => [{ label: 'Show server log', run: () => {} }]);
    expect(rows(policy.buildTileMenuItems('jupyter'))).toEqual([
      'This panel',
      'Show server log',
      'Tile',
      'Open in a new tile',
      'Open in new window',
      'Reload',
      'Help',
      'Remove from rail',
      'Reset layout',
    ]);
    // No docs page for the panel, no Help.
    expect(rows(policy.buildTileMenuItems('tuning'))).not.toContain('Help');
  });

  test('in the pane’s own window: Reattach, Reload, Help and the rail row, nothing that acts on the dock', () => {
    document.documentElement.dataset.docsUrl = 'https://docs.example/';
    paneWindow.isDetached.mockReturnValue(true);
    policy.setServiceMenuExtras('jupyter', () => [{ label: 'Show server log', run: () => {} }]);
    expect(rows(policy.buildTileMenuItems('jupyter', { mode: 'detached' }))).toEqual([
      'This panel',
      'Show server log',
      'Tile',
      'Reattach',
      'Reload',
      'Help',
      'Remove from rail',
    ]);
  });

  test('the ⋯ button hangs the menu under itself; a second press closes it', () => {
    const anchorEl = document.createElement('div');
    const triggerEl = document.createElement('button');
    anchorEl.append(triggerEl);
    document.body.append(anchorEl);
    const opts = { x: 0, y: 0, anchorEl, triggerEl, mode: /** @type {const} */ ('tile') };
    expect(policy.openTileContextMenu('ariel', opts)).toBe(true);
    expect(triggerEl.getAttribute('aria-expanded')).toBe('true');
    expect(menuLabels()).toContain('Reload');
    policy.openTileContextMenu('ariel', opts);
    expect(document.querySelector('.rail-context-menu')).toBeNull();
    expect(triggerEl.getAttribute('aria-expanded')).toBe('false');
  });
});
