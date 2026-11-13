/**
 * The Agent's two menus (panel-menu-policy.js): the tile header — section
 * Agent (Switch to terminal / chat, Export conversation, Desktop
 * notifications), section Tile (Open in new window, Reveal in sidebar, Close
 * Agent pane, Restart terminal on the terminal) — and the
 * Agent row in the sidebar — the head of the session tree — gets the tree's
 * menu (New session, Artifacts by, Show pinned, Collapse/Expand all), New session
 * through switchTo.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-menu-policy.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const surface = vi.hoisted(() => ({
  terminal: true,
  showChatSurface: vi.fn(),
  showTerminalSurface: vi.fn(),
}));
const dock = vi.hoisted(() => ({
  openTerminalPanel: vi.fn(),
  closeTerminalPanel: vi.fn(),
  isTerminalPanelOpen: vi.fn(() => true),
  resetDockLayout: vi.fn(),
  closeDockPanel: vi.fn(() => true),
  maximizeTile: vi.fn(() => true),
  restoreTile: vi.fn(),
  maximizedTileKey: vi.fn(() => /** @type {string | null} */ (null)),
}));
// The tile × wraps its close in the echo guard; Close tile must run the same.
const echo = vi.hoisted(() => ({
  depth: 0,
  /** @type {string[]} */
  log: [],
}));
const paneWindow = vi.hoisted(() => ({
  detach: vi.fn(),
  detachBlocker: vi.fn(() => /** @type {string | null} */ (null)),
  focusWindow: vi.fn(),
  reattach: vi.fn(),
  closeDetached: vi.fn(() => true),
  isDetached: vi.fn(() => false),
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
  showChatSurface: surface.showChatSurface,
  showTerminalSurface: surface.showTerminalSurface,
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
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-sync.js', async (original) => ({
  .../** @type {object} */ (await original()),
  withEchoSuppressed: (/** @type {() => unknown} */ fn) => {
    echo.depth += 1;
    try {
      return fn();
    } finally {
      echo.depth -= 1;
    }
  },
}));
const frames = vi.hoisted(() => ({ reloadPanel: vi.fn(() => true) }));
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js',
  async (original) => ({ .../** @type {object} */ (await original()), ...frames })
);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/session-model.js',
  () => sessionModel
);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sessions.js', () => sessions);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js', () => tree);
// Reveal in sidebar drives the sidebar's two real mutators; neither is
// initialised in this file, so their calls are recorded instead.
const reveal = vi.hoisted(() => ({
  collapsed: false,
  setSidebarCollapsed: vi.fn(),
  setGroupFolded: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar.js', async (original) => ({
  .../** @type {object} */ (await original()),
  isSidebarCollapsed: () => reveal.collapsed,
  setSidebarCollapsed: reveal.setSidebarCollapsed,
}));
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-fold.js',
  async (original) => ({
    .../** @type {object} */ (await original()),
    setGroupFolded: reveal.setGroupFolded,
  })
);

// The tile bar's Open panel ▸ rows place through panel-placement's verb; the
// verb's own behaviour is panel-placement.test.mjs's contract.
const placing = vi.hoisted(() => ({
  openPanelAtTile: vi.fn(() => null),
  dockedTileOf: vi.fn(
    /** @param {string} id @returns {string | null} */ (id) => (id ? 'g7' : null)
  ),
}));
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/panel-placement.js',
  async (original) => ({ .../** @type {object} */ (await original()), ...placing })
);

const policy = await import(`${JS}/panel-menu-policy.js`);
const sidebar = await import(`${JS}/sidebar.js`);

/** @typedef {{label?: string, run?: () => unknown}} Item */
const { TERMINAL_RAIL_ID } = await import(`${JS}/panel-catalog.js`);
const { groupOf } = await import(`${JS}/sidebar-groups.js`);

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
  delete document.documentElement.dataset.terminal;
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
  test('on the terminal surface: Agent (Switch to chat), then Tile — Maximize, Open in new window, Reveal in sidebar, Close Agent pane, Restart terminal; no New session, no Reset layout', () => {
    const anchorEl = document.createElement('div');
    document.body.append(anchorEl);
    expect(policy.openTileContextMenu(TERMINAL_RAIL_ID, { x: 1, y: 1, anchorEl })).toBe(true);
    expect(menuLabels()).toEqual([
      'Switch to chat',
      'Maximize',
      'Zoom',
      'Open in new window',
      'Reveal in sidebar',
      'Close Agent pane',
      'Restart terminal',
    ]);
    expect(document.querySelector('.rail-context-menu')?.getAttribute('aria-label')).toBe(
      'Agent actions'
    );
  });

  test("in chat the PTY verb is withheld, and the section reads Agent: Switch to terminal, then the chat's rows", () => {
    surface.terminal = false;
    const run = vi.fn();
    policy.setServiceMenuExtras(TERMINAL_RAIL_ID, () => [
      { label: 'Export conversation', run },
      { label: 'Desktop notifications', run: () => {} },
    ]);
    /** @type {any[]} */
    const items = policy.buildTerminalMenuItems();
    expect(items.map((i) => i.heading ?? i.label ?? '—')).toEqual([
      'Agent',
      'Switch to terminal',
      'Export conversation',
      'Desktop notifications',
      'Tile',
      'Maximize',
      'Zoom',
      'Open in new window',
      'Reveal in sidebar',
      '—',
      'Close Agent pane',
    ]);
    items.find((i) => i.label === 'Export conversation')?.run?.();
    expect(run).toHaveBeenCalledTimes(1);
    policy.setServiceMenuExtras(TERMINAL_RAIL_ID, null);
  });

  test('Switch runs the session switcher’s handlers, and is absent without a terminal', () => {
    /** @type {any[]} */
    let items = policy.buildTerminalMenuItems();
    items.find((i) => i.label === 'Switch to chat')?.run?.();
    expect(surface.showChatSurface).toHaveBeenCalledTimes(1);
    surface.terminal = false;
    items = policy.buildTerminalMenuItems();
    items.find((i) => i.label === 'Switch to terminal')?.run?.();
    expect(surface.showTerminalSurface).toHaveBeenCalledTimes(1);
    document.documentElement.dataset.terminal = 'off';
    items = policy.buildTerminalMenuItems();
    expect(items.map((i) => i.heading ?? i.label).filter(Boolean)).toEqual([
      'Maximize',
      'Zoom',
      'Open in new window',
      'Reveal in sidebar',
      'Close Agent pane',
    ]);
  });

  test('the destructive rows go last; Restart terminal only on the terminal surface', () => {
    /** @type {any[]} */
    const items = policy.buildTerminalMenuItems();
    const tail = items.slice(-3);
    expect(tail.map((i) => i.label ?? (i.divider ? '—' : ''))).toEqual([
      '—',
      'Close Agent pane',
      'Restart terminal',
    ]);
    expect(tail[1].danger && tail[2].danger).toBe(true);
  });

  test('Reveal in sidebar reveals the Sessions row', () => {
    /** @type {any[]} */
    const items = policy.buildTerminalMenuItems();
    items.find((i) => i.label === 'Reveal in sidebar')?.run?.();
    expect(reveal.setGroupFolded).toHaveBeenCalledWith(groupOf(TERMINAL_RAIL_ID), false);
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
    items.find((i) => i.label === 'Close Agent pane')?.run?.();
    expect(dock.resetDockLayout).not.toHaveBeenCalled();
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
    expect(menuLabels()).toEqual(['New session', 'Artifacts by', 'Show pinned sessions', 'Expand all']);
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
    expect(menuLabels()).toEqual([
      'New session',
      'Artifacts by',
      'Show pinned sessions',
      'Collapse all',
    ]);
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
  const rows = (/** @type {any[]} */ items) =>
    items.map((i) => i.heading ?? i.label ?? (i.divider ? '—' : ''));

  test('Agent holds the registered rows alone; Tile is Reattach, then Close window', () => {
    paneWindow.isDetached.mockReturnValue(true);
    policy.setServiceMenuExtras(TERMINAL_RAIL_ID, () => [
      { label: 'Export conversation', run: () => {} },
      { label: 'Desktop notifications', run: () => {} },
    ]);
    for (const terminal of [true, false]) {
      surface.terminal = terminal;
      expect(rows(policy.buildTerminalMenuItems({ mode: 'detached' }))).toEqual([
        'Agent',
        'Export conversation',
        'Desktop notifications',
        'Tile',
        'Reattach',
        'Zoom',
        '—',
        'Close window',
      ]);
    }
    policy.setServiceMenuExtras(TERMINAL_RAIL_ID, null);
    paneWindow.isDetached.mockReturnValue(false);
  });

  test('with no registered rows the bar is the Tile rows alone', () => {
    paneWindow.isDetached.mockReturnValue(true);
    expect(rows(policy.buildTerminalMenuItems({ mode: 'detached' }))).toEqual([
      'Reattach',
      'Zoom',
      '—',
      'Close window',
    ]);
    paneWindow.isDetached.mockReturnValue(false);
  });

  test('Close window is the window’s own close: the pane closes, the panel stays on the rail', () => {
    paneWindow.isDetached.mockReturnValue(true);
    /** @type {any[]} */
    const items = policy.buildTerminalMenuItems({ mode: 'detached' });
    items.find((i) => i.label === 'Close window')?.run?.();
    expect(paneWindow.closeDetached).toHaveBeenCalledExactlyOnceWith(TERMINAL_RAIL_ID);
    expect(dock.closeTerminalPanel).not.toHaveBeenCalled();
    paneWindow.isDetached.mockReturnValue(false);
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

  test('docked: the Tile verbs alone, Reveal in sidebar for a member, with no heading, Close tile last behind a divider', () => {
    const items = policy.buildTileMenuItems('ariel');
    expect(rows(items)).toEqual([
      'Open panel',
      'Maximize',
      'Zoom',
      'Open in new window',
      'Reload',
      'Reveal in sidebar',
      'Close tile',
    ]);
    expect(items.map((/** @type {any} */ i) => (i.divider ? '—' : i.label))).toEqual([
      'Open panel',
      'Maximize',
      'Zoom',
      'Open in new window',
      'Reload',
      'Reveal in sidebar',
      '—',
      'Close tile',
    ]);
    expect(items.at(-1)).toMatchObject({ label: 'Close tile', glyph: '×', danger: true });
    items.find((/** @type {any} */ i) => i.label === 'Reload').run();
    expect(frames.reloadPanel).toHaveBeenCalledExactlyOnceWith('ariel');
  });

  test('no rail or layout verbs on a service tile bar: no Move to, Remove from rail, Reset layout, Open in a new tile', () => {
    const labels = policy
      .buildTileMenuItems('ariel')
      .map((/** @type {any} */ i) => i.label)
      .filter(Boolean);
    for (const gone of ['Remove from rail', 'Reset layout', 'Open in a new tile', 'Pin to rail']) {
      expect(labels).not.toContain(gone);
    }
    expect(labels.some((/** @type {string} */ l) => l.startsWith('Move to'))).toBe(false);
  });

  test('Close tile runs the tile ×: closeDockPanel for the panel, inside the echo guard', () => {
    /** @type {number[]} */
    const depths = [];
    dock.closeDockPanel.mockImplementationOnce(() => {
      depths.push(echo.depth);
      return true;
    });
    const close = policy
      .buildTileMenuItems('ariel')
      .find((/** @type {any} */ i) => i.label === 'Close tile');
    close.run();
    expect(dock.closeDockPanel).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(depths).toEqual([1]);
    expect(dock.resetDockLayout).not.toHaveBeenCalled();
  });

  test('an off-rail tile offers Pin to rail before Close tile', () => {
    policy.initMenuPolicy({
      getRailEl: () => rail,
      isMember: () => false,
      getActiveTabId: () => null,
      activateTab: () => {},
      showPanel: () => {},
      retireTile: () => {},
      labelOf: (/** @type {string} */ id) => id,
      retryStart: () => {},
    });
    expect(rows(policy.buildTileMenuItems('ariel'))).toEqual([
      'Open panel',
      'Maximize',
      'Zoom',
      'Open in new window',
      'Reload',
      'Pin to rail',
      'Close tile',
    ]);
    // Pin to rail takes Reveal's slot: an off-rail panel has no row to reveal.
    expect(rows(policy.buildTileMenuItems('ariel'))).not.toContain('Reveal in sidebar');
  });

  test('an artifact tile is no panel the rail can hold: no Pin to rail', () => {
    policy.initMenuPolicy({
      getRailEl: () => rail,
      isMember: () => false,
      getActiveTabId: () => null,
      activateTab: () => {},
      showPanel: () => {},
      retireTile: () => {},
      labelOf: (/** @type {string} */ id) => id,
      retryStart: () => {},
    });
    expect(rows(policy.buildTileMenuItems('artifact:a1'))).toEqual([
      'Open panel',
      'Maximize',
      'Zoom',
      'Open in new window',
      'Reload',
      'Close tile',
    ]);
  });

  test("the panel's own rows head the menu as This panel, and Help shows with a docs page", () => {
    document.documentElement.dataset.docsUrl = 'https://docs.example/';
    policy.setServiceMenuExtras('jupyter', () => [{ label: 'Show server log', run: () => {} }]);
    expect(rows(policy.buildTileMenuItems('jupyter'))).toEqual([
      'This panel',
      'Show server log',
      'Tile',
      'Open panel',
      'Maximize',
      'Zoom',
      'Open in new window',
      'Reload',
      'Help',
      'Reveal in sidebar',
      'Close tile',
    ]);
    // No docs page for the panel, no Help.
    expect(rows(policy.buildTileMenuItems('tuning'))).not.toContain('Help');
  });

  test('the no-dockview fallback bar omits every dock verb', () => {
    document.documentElement.dataset.docsUrl = 'https://docs.example/';
    policy.setServiceMenuExtras('jupyter', () => [{ label: 'Show server log', run: () => {} }]);
    expect(rows(policy.buildTileMenuItems('jupyter', { mode: 'fallback' }))).toEqual([
      'This panel',
      'Show server log',
      'Tile',
      'Reload',
      'Help',
    ]);
  });

  test('in the pane’s own window: This panel as docked, then Reattach, Reload, Help, Close window — nothing that acts on the dock or the rail', () => {
    document.documentElement.dataset.docsUrl = 'https://docs.example/';
    paneWindow.isDetached.mockReturnValue(true);
    policy.setServiceMenuExtras('jupyter', () => [{ label: 'Show server log', run: () => {} }]);
    /** @type {any[]} */
    const items = policy.buildTileMenuItems('jupyter', { mode: 'detached' });
    expect(items.map((i) => i.heading ?? i.label ?? (i.divider ? '—' : ''))).toEqual([
      'This panel',
      'Show server log',
      'Tile',
      'Reattach',
      'Zoom',
      'Reload',
      'Help',
      '—',
      'Close window',
    ]);
    items.find((i) => i.label === 'Close window')?.run?.();
    expect(paneWindow.closeDetached).toHaveBeenCalledExactlyOnceWith('jupyter');
    paneWindow.isDetached.mockReturnValue(false);
  });

  test('an off-rail panel’s window offers Pin to rail before Close window', () => {
    delete document.documentElement.dataset.docsUrl;
    paneWindow.isDetached.mockReturnValue(true);
    policy.initMenuPolicy({
      getRailEl: () => rail,
      isMember: () => false,
      getActiveTabId: () => null,
      activateTab: () => {},
      showPanel: () => {},
      retireTile: () => {},
      labelOf: (/** @type {string} */ id) => id,
      retryStart: () => {},
    });
    /** @type {any[]} */
    const items = policy.buildTileMenuItems('ariel', { mode: 'detached' });
    expect(items.map((i) => i.heading ?? i.label ?? (i.divider ? '—' : ''))).toEqual([
      'Reattach',
      'Zoom',
      'Reload',
      'Pin to rail',
      '—',
      'Close window',
    ]);
    paneWindow.isDetached.mockReturnValue(false);
  });

  describe('Open panel ▸ — the Tile section’s first row', () => {
    /** @type {HTMLElement | null} */
    let panelRail = null;

    /**
     * Put a #panel-rail in the page with one row per `[id, label, cls?]`, the
     * rows untiledPanels (dock-menus.js) lists.
     * @param {Array<[string, string, string?]>} entries
     */
    function railOf(entries) {
      panelRail = document.createElement('ul');
      panelRail.id = 'panel-rail';
      for (const [id, label, cls] of entries) {
        const btn = document.createElement('button');
        btn.className = `panel-rail-button${cls ? ` ${cls}` : ''}`;
        btn.dataset.panelId = id;
        const span = document.createElement('span');
        span.className = 'osprey-nav-label';
        span.textContent = label;
        btn.append(span);
        panelRail.append(btn);
      }
      document.body.append(panelRail);
    }

    afterEach(() => {
      panelRail?.remove();
      panelRail = null;
    });

    /** @param {any[]} items @returns {any} */
    const openPanel = (items) => items.find((i) => i.label === 'Open panel');

    test('it heads the docked Tile section with the placements Right, Down, Replace here', () => {
      railOf([
        ['tuning', 'Tuning'],
        ['logbook', 'Logbook'],
      ]);
      const row = openPanel(policy.buildTileMenuItems('ariel'));
      expect(policy.buildTileMenuItems('ariel')[0].label).toBe('Open panel');
      expect(row.submenu.map((/** @type {any} */ i) => i.label)).toEqual([
        'Right',
        'Down',
        'Replace here',
      ]);
      for (const placement of row.submenu) {
        expect(placement.submenu.map((/** @type {any} */ i) => i.label)).toEqual([
          'Tuning',
          'Logbook',
        ]);
      }
    });

    test('each placement lists the untiled panels; a panel that cannot open is disabled with its reason', () => {
      railOf([
        ['tuning', 'Tuning'],
        ['logbook', 'Logbook', 'disabled'],
      ]);
      const right = openPanel(policy.buildTileMenuItems('ariel')).submenu[0];
      expect(right.submenu[0]).not.toHaveProperty('disabled');
      expect(right.submenu[1]).toMatchObject({ label: 'Logbook', disabled: true });
      expect(typeof right.submenu[1].tip).toBe('string');
    });

    test('with nothing left to open, every placement shows one disabled row', () => {
      const row = openPanel(policy.buildTileMenuItems('ariel'));
      for (const placement of row.submenu) {
        expect(placement.submenu).toEqual([{ label: 'No panels to open', disabled: true }]);
      }
    });

    test('Right, Down and Replace here place the panel against THIS tile through the placement verb', () => {
      railOf([['tuning', 'Tuning']]);
      placing.dockedTileOf.mockImplementation((id) => (id === 'ariel' ? 's2' : 's1'));
      const [right, down, replace] = openPanel(policy.buildTileMenuItems('ariel')).submenu;
      right.submenu[0].run();
      down.submenu[0].run();
      replace.submenu[0].run();
      expect(placing.dockedTileOf).toHaveBeenCalledWith('ariel');
      expect(placing.openPanelAtTile.mock.calls).toEqual([
        ['tuning', 's2', 'right'],
        ['tuning', 's2', 'down'],
        ['tuning', 's2', 'replace'],
      ]);
      placing.dockedTileOf.mockImplementation(() => 'g7');
    });

    test('a bar whose panel holds no tile places nothing', () => {
      railOf([['tuning', 'Tuning']]);
      placing.dockedTileOf.mockImplementation(() => null);
      openPanel(policy.buildTileMenuItems('ariel')).submenu[0].submenu[0].run();
      expect(placing.openPanelAtTile).not.toHaveBeenCalled();
      placing.dockedTileOf.mockImplementation(() => 'g7');
    });

    test('not on pop-out bars, the fallback bar, or the Agent tile', () => {
      railOf([['tuning', 'Tuning']]);
      paneWindow.isDetached.mockReturnValue(true);
      expect(openPanel(policy.buildTileMenuItems('ariel', { mode: 'detached' }))).toBeUndefined();
      paneWindow.isDetached.mockReturnValue(false);
      expect(openPanel(policy.buildTileMenuItems('ariel', { mode: 'fallback' }))).toBeUndefined();
      expect(openPanel(policy.buildTerminalMenuItems())).toBeUndefined();
    });
  });

  test('⋯ and right-click open the same rows', () => {
    const anchorEl = document.createElement('div');
    const triggerEl = document.createElement('button');
    anchorEl.append(triggerEl);
    document.body.append(anchorEl);
    policy.setServiceMenuExtras('jupyter', () => [{ label: 'Show server log', run: () => {} }]);
    const tile = /** @type {const} */ ('tile');
    expect(policy.openTileContextMenu('jupyter', { x: 5, y: 5, anchorEl, mode: tile })).toBe(true);
    const viaRightClick = menuLabels();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(document.querySelector('.rail-context-menu')).toBeNull();
    expect(
      policy.openTileContextMenu('jupyter', { x: 0, y: 0, anchorEl, triggerEl, mode: tile })
    ).toBe(true);
    expect(menuLabels()).toEqual(viaRightClick);
    expect(viaRightClick).toEqual([
      'Show server log',
      'Open panel',
      'Maximize',
      'Zoom',
      'Open in new window',
      'Reload',
      'Reveal in sidebar',
      'Close tile',
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

describe('Maximize / Restore on the tile bars', () => {
  /** @param {any[]} items */
  const labels = (items) => items.map((i) => i.heading ?? i.label ?? (i.divider ? '—' : ''));

  afterEach(() => {
    dock.maximizedTileKey.mockReturnValue(null);
    paneWindow.isDetached.mockReturnValue(false);
  });

  test('the service bar carries Maximize right after Open panel ▸, and it maximizes THIS tile', () => {
    /** @type {any[]} */
    const items = policy.buildTileMenuItems('ariel');
    const l = labels(items);
    expect(l.indexOf('Maximize')).toBe(l.indexOf('Open panel') + 1);
    expect(l).not.toContain('Restore');
    items.find((i) => i.label === 'Maximize')?.run?.();
    expect(dock.maximizeTile).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(dock.restoreTile).not.toHaveBeenCalled();
  });

  test('the Agent bar carries Maximize first in its Tile section, maximizing the Agent tile', () => {
    /** @type {any[]} */
    const items = policy.buildTerminalMenuItems();
    const l = labels(items);
    expect(l[l.indexOf('Tile') + 1]).toBe('Maximize');
    items.find((i) => i.label === 'Maximize')?.run?.();
    expect(dock.maximizeTile).toHaveBeenCalledExactlyOnceWith(TERMINAL_RAIL_ID);
  });

  test('while a tile is maximized the row reads Restore, in the same slot, and restores', () => {
    dock.maximizedTileKey.mockReturnValue('jupyter');
    /** @type {any[]} */
    const service = policy.buildTileMenuItems('ariel');
    const l = labels(service);
    expect(l).not.toContain('Maximize');
    expect(l.indexOf('Restore')).toBe(l.indexOf('Open panel') + 1);
    service.find((i) => i.label === 'Restore')?.run?.();
    expect(dock.restoreTile).toHaveBeenCalledTimes(1);
    expect(dock.maximizeTile).not.toHaveBeenCalled();
    /** @type {any[]} */
    const agent = policy.buildTerminalMenuItems();
    const a = labels(agent);
    expect(a[a.indexOf('Tile') + 1]).toBe('Restore');
    agent.find((i) => i.label === 'Restore')?.run?.();
    expect(dock.restoreTile).toHaveBeenCalledTimes(2);
  });

  test('the label is read live, each time the menu opens', () => {
    const anchorEl = document.createElement('div');
    document.body.append(anchorEl);
    const opts = { x: 0, y: 0, anchorEl, mode: /** @type {const} */ ('tile') };
    policy.openTileContextMenu('ariel', opts);
    expect(menuLabels()).toContain('Maximize');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    dock.maximizedTileKey.mockReturnValue('ariel');
    policy.openTileContextMenu('ariel', opts);
    expect(menuLabels()).toContain('Restore');
    expect(menuLabels()).not.toContain('Maximize');
  });

  test('not on pop-out bars or the no-dockview fallback bar, maximized or not', () => {
    for (const key of [null, 'ariel']) {
      dock.maximizedTileKey.mockReturnValue(key);
      paneWindow.isDetached.mockReturnValue(true);
      const detached = /** @type {const} */ ('detached');
      const fallback = /** @type {const} */ ('fallback');
      for (const items of [
        policy.buildTileMenuItems('ariel', { mode: detached }),
        policy.buildTerminalMenuItems({ mode: detached }),
      ]) {
        expect(labels(items)).not.toContain('Maximize');
        expect(labels(items)).not.toContain('Restore');
      }
      paneWindow.isDetached.mockReturnValue(false);
      for (const items of [
        policy.buildTileMenuItems('ariel', { mode: fallback }),
        policy.buildTerminalMenuItems({ mode: fallback }),
      ]) {
        expect(labels(items)).not.toContain('Maximize');
        expect(labels(items)).not.toContain('Restore');
      }
    }
  });
});

describe('revealInSidebar', () => {
  /** @param {string} id */
  const addRow = (id) => {
    const li = document.createElement('li');
    li.className = 'rail-entry';
    const btn = document.createElement('button');
    btn.className = 'panel-rail-button';
    btn.dataset.panelId = id;
    li.append(btn);
    rail.append(li);
    return btn;
  };
  /** @type {any} */
  let scroll;
  // happy-dom's own scrollIntoView, put back after each test: the worker can
  // share Element.prototype with the test files that run after this one.
  const nativeScroll = Object.getOwnPropertyDescriptor(Element.prototype, 'scrollIntoView');

  beforeEach(() => {
    reveal.collapsed = false;
    scroll = vi.fn();
    /** @type {any} */ (Element.prototype).scrollIntoView = scroll;
  });
  afterEach(() => {
    reveal.collapsed = false;
    if (nativeScroll) Object.defineProperty(Element.prototype, 'scrollIntoView', nativeScroll);
    else delete (/** @type {any} */ (Element.prototype).scrollIntoView);
  });

  test('a collapsed sidebar and a folded group: expands both, then scrolls to and flashes the row', () => {
    reveal.collapsed = true;
    const btn = addRow('ariel');
    /** @type {string[]} */
    const order = [];
    reveal.setSidebarCollapsed.mockImplementationOnce(() => order.push('expand'));
    reveal.setGroupFolded.mockImplementationOnce(() => order.push('unfold'));
    scroll.mockImplementationOnce(() => order.push('scroll'));
    expect(policy.revealInSidebar('ariel')).toBe(true);
    expect(reveal.setSidebarCollapsed).toHaveBeenCalledExactlyOnceWith(false);
    expect(reveal.setGroupFolded).toHaveBeenCalledExactlyOnceWith(groupOf('ariel'), false);
    expect(order).toEqual(['expand', 'unfold', 'scroll']);
    expect(scroll.mock.contexts[0]).toBe(btn);
    expect(btn.classList.contains('agent-flash')).toBe(true);
  });

  test('an expanded sidebar is left alone; the scroll never animates (reduced motion)', () => {
    addRow('ariel');
    policy.revealInSidebar('ariel');
    expect(reveal.setSidebarCollapsed).not.toHaveBeenCalled();
    expect(scroll).toHaveBeenCalledOnce();
    expect(scroll.mock.calls[0][0]?.behavior).not.toBe('smooth');
  });

  test('the Agent pane reveals the Sessions row', () => {
    const sessionsRow = /** @type {HTMLElement} */ (
      rail.querySelector(`[data-panel-id="${TERMINAL_RAIL_ID}"]`)
    );
    expect(policy.revealInSidebar(TERMINAL_RAIL_ID)).toBe(true);
    expect(reveal.setGroupFolded).toHaveBeenCalledExactlyOnceWith(groupOf(TERMINAL_RAIL_ID), false);
    expect(scroll.mock.contexts[0]).toBe(sessionsRow);
    expect(sessionsRow.classList.contains('agent-flash')).toBe(true);
  });

  test('a panel with no row in view reports false and flashes nothing', () => {
    expect(policy.revealInSidebar('nowhere')).toBe(false);
    expect(scroll).not.toHaveBeenCalled();
  });

  test('the tile bar row runs it', () => {
    const btn = addRow('ariel');
    const row = policy
      .buildTileMenuItems('ariel')
      .find((/** @type {any} */ i) => i.label === 'Reveal in sidebar');
    row.run();
    expect(btn.classList.contains('agent-flash')).toBe(true);
  });
});

describe('focusTile', () => {
  afterEach(() => paneWindow.isDetached.mockReturnValue(false));

  /** @param {(id: string, o?: object) => void} activateTab @param {string | null} [active] */
  const init = (activateTab, active = null, member = true) =>
    policy.initMenuPolicy({
      getRailEl: () => rail,
      isMember: () => member,
      getActiveTabId: () => active,
      activateTab,
      showPanel: () => {},
      retireTile: vi.fn(),
      labelOf: (/** @type {string} */ id) => id,
      retryStart: () => {},
    });

  test('a docked member tile is activated as a user focus', () => {
    const activateTab = vi.fn();
    init(activateTab);
    policy.focusTile('ariel');
    expect(activateTab).toHaveBeenCalledExactlyOnceWith('ariel', { userInitiated: true });
    expect(paneWindow.focusWindow).not.toHaveBeenCalled();
  });

  test('the already-active tile is focused, never retired', () => {
    const activateTab = vi.fn();
    init(activateTab, 'ariel');
    policy.focusTile('ariel');
    expect(activateTab).toHaveBeenCalledOnce();
  });

  test('an off-rail tile is focused without the member report', () => {
    const activateTab = vi.fn();
    init(activateTab, null, false);
    policy.focusTile('ariel');
    expect(activateTab).toHaveBeenCalledExactlyOnceWith('ariel', { userInitiated: false });
  });

  test('a detached panel brings its window to the front', () => {
    const activateTab = vi.fn();
    init(activateTab);
    paneWindow.isDetached.mockReturnValue(true);
    policy.focusTile('ariel');
    expect(paneWindow.focusWindow).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(activateTab).not.toHaveBeenCalled();
  });

  test('the Agent pane focuses through openTerminalPanel, or its window when detached', () => {
    const activateTab = vi.fn();
    init(activateTab);
    policy.focusTile(TERMINAL_RAIL_ID);
    expect(dock.openTerminalPanel).toHaveBeenCalledOnce();
    paneWindow.isDetached.mockReturnValue(true);
    policy.focusTile(TERMINAL_RAIL_ID);
    expect(paneWindow.focusWindow).toHaveBeenCalledExactlyOnceWith(TERMINAL_RAIL_ID);
    expect(activateTab).not.toHaveBeenCalled();
  });
});

describe('a service rail entry, by state', () => {
  /** @type {ReturnType<typeof vi.fn>} */
  let retryStart;
  /** @type {HTMLElement} */
  let entryBtn;

  /** @param {{ active?: string | null }} [o] */
  const init = ({ active = null } = {}) =>
    policy.initMenuPolicy({
      getRailEl: () => rail,
      isMember: () => true,
      getActiveTabId: () => active,
      activateTab: () => {},
      showPanel: () => {},
      retireTile: () => {},
      labelOf: (/** @type {string} */ id) => id,
      retryStart,
    });

  /** Row labels in order: a caption as `# <heading>`, a divider as `—`. */
  const labels = (/** @type {any[]} */ items) =>
    items.map((i) => i.label ?? (i.heading ? `# ${i.heading}` : '—'));

  beforeEach(() => {
    retryStart = vi.fn();
    const row = document.createElement('li');
    row.className = 'rail-entry';
    row.dataset.id = 'svc';
    entryBtn = document.createElement('button');
    entryBtn.className = 'panel-rail-button';
    entryBtn.dataset.panelId = 'svc';
    row.append(entryBtn);
    rail.append(row);
    init();
  });
  afterEach(() => {
    paneWindow.isDetached.mockReturnValue(false);
    policy.setServiceMenuExtras('svc', null);
  });

  test('docked (the surfaced panel without a dock): Focus tile · Open in a new tile · Open in new window', () => {
    init({ active: 'svc' });
    expect(labels(policy.buildServiceMenuItems('svc'))).toEqual([
      'Focus tile',
      'Open in a new tile',
      'Open in new window',
      'Move to',
      '—',
      'Remove from rail',
    ]);
  });

  test('in its window: Show window · Reattach', () => {
    paneWindow.isDetached.mockReturnValue(true);
    const items = policy.buildServiceMenuItems('svc');
    expect(labels(items)).toEqual(['Show window', 'Reattach', 'Move to', '—', 'Remove from rail']);
    items[0].run();
    expect(paneWindow.focusWindow).toHaveBeenCalledExactlyOnceWith('svc');
  });

  test('tile closed: Open · Open in new window', () => {
    expect(labels(policy.buildServiceMenuItems('svc'))).toEqual([
      'Open',
      'Open in new window',
      'Move to',
      '—',
      'Remove from rail',
    ]);
  });

  test('Move to is one submenu row; New group… ends it', () => {
    const move = policy
      .buildServiceMenuItems('svc')
      .find((/** @type {any} */ i) => i.label === 'Move to');
    expect(move?.run).toBeUndefined();
    const sub = /** @type {any[]} */ (move?.submenu);
    expect(sub.at(-1).label).toBe('New group…');
    expect(sub.some((i) => i.label === groupOf('svc'))).toBe(false);
  });

  test('a panel’s own rows come first, under This panel, danger last within the section', () => {
    policy.setServiceMenuExtras('svc', () => [
      { label: 'Restart kernel server', danger: true, run: () => {} },
      { label: 'Refresh', run: () => {} },
    ]);
    expect(labels(policy.buildServiceMenuItems('svc'))).toEqual([
      '# This panel',
      'Refresh',
      '—',
      'Restart kernel server',
      '—',
      'Open',
      'Open in new window',
      'Move to',
      '—',
      'Remove from rail',
    ]);
  });

  test('failed: Retry start · Remove from rail, and Retry start is the click’s retry', () => {
    entryBtn.closest('li')?.classList.add('failed');
    entryBtn.classList.add('failed');
    const items = policy.buildServiceMenuItems('svc');
    expect(labels(items)).toEqual(['Retry start', '—', 'Remove from rail']);
    items[0].run();
    expect(retryStart).toHaveBeenCalledExactlyOnceWith('svc');
    expect(policy.openRailContextMenu('svc', 1, 1)).toBe(true);
  });

  test('never started: no menu, so the browser keeps its own', () => {
    entryBtn.closest('li')?.classList.add('disabled');
    entryBtn.classList.add('disabled');
    expect(policy.openRailContextMenu('svc', 1, 1)).toBe(false);
    expect(document.querySelector('.rail-context-item')).toBeNull();
  });
});
