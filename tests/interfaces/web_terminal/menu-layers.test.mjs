/**
 * The shell menus' Layer rule, checked across every place at once.
 *
 * Every shell place's rows are built in representative states — the Sidebar
 * places (rail entry docked / closed / detached / off-rail / failed, group
 * heading, Sessions row with the Agent pane tiled / tileless / detached,
 * session, run and artifact rows, pinned heading, tree group), the Workspace
 * places (sidebar header, empty sidebar, empty dock slot), the Tile places
 * (service tile bar docked / off-rail / fallback / pop-out, Agent tile bar on
 * the terminal / chat / fallback / pop-out) and the Content places (chat
 * message, chat background, terminal body). Three things are asserted:
 *
 *   1. The proposal's Layer rule as written: dock verbs only at Tile places
 *      and the empty dock slot, listing verbs only at Sidebar places and the
 *      Workspace's sidebar places, and only the named exceptions (the rail
 *      entry's open verbs, the Sessions row's tileless Agent pane rows, Pin to
 *      rail on an off-rail tile bar). Reset layout is Workspace-only.
 *   2. Every verb the GP7 base menus offered is still reachable at its place
 *      and still calls the handler it called there (a spy, or the handler's
 *      observable effect where the handler is module-internal).
 *   3. Destructive rows sit last, behind a divider, in every shell menu.
 *
 * Maximize / Restore are classified as dock verbs from the rule; the test does
 * not require the row to exist.
 *
 *   npx vitest run tests/interfaces/web_terminal/menu-layers.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

// ---- Mocks: every handler a menu row reaches is a spy ----

const captured = vi.hoisted(() => ({
  /** @type {any[]} */ menus: [],
  /** @type {any[]} */ places: [],
}));
const surface = vi.hoisted(() => ({
  terminal: true,
  showChatSurface: vi.fn(),
  showTerminalSurface: vi.fn(),
}));
const dock = vi.hoisted(() => ({
  open: true,
  openTerminalPanel: vi.fn(),
  closeTerminalPanel: vi.fn(() => true),
  isTerminalPanelOpen: vi.fn(() => true),
  resetDockLayout: vi.fn(),
  closeDockPanel: vi.fn(() => true),
  maximizeTile: vi.fn(() => true),
  restoreTile: vi.fn(),
  maximizedTileKey: vi.fn(() => /** @type {string | null} */ (null)),
  onMaximizeChange: vi.fn(() => () => {}),
}));
const paneWindow = vi.hoisted(() => ({
  detached: new Set(),
  detach: vi.fn(),
  detachBlocker: vi.fn(() => null),
  focusWindow: vi.fn(),
  reattach: vi.fn(),
  closeDetached: vi.fn(() => true),
  isDetached: vi.fn((/** @type {string} */ id) => paneWindow.detached.has(id)),
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
  forgetSession: vi.fn(),
  setPinned: vi.fn(async () => true),
}));
const tree = vi.hoisted(() => ({
  setTreeView: vi.fn(),
  toggleAllSessions: vi.fn(),
  openInAgent: vi.fn(async () => {}),
}));
const listing = vi.hoisted(() => ({
  ARTIFACTS_BASE: '/panel/artifacts',
  fetchArtifact: vi.fn(async () => null),
  heldEntry: vi.fn(() => null),
  heldRun: vi.fn(() => null),
  fetchRun: vi.fn(async () => null),
  readTreeView: vi.fn(() => 'kind'),
}));
const confirm = vi.hoisted(() => ({ showConfirm: vi.fn() }));
const frames = vi.hoisted(() => ({ reloadPanel: vi.fn(() => true) }));
const help = vi.hoisted(() => ({
  helpUrl: vi.fn((/** @type {string} */ id) => (id === 'svc' ? 'https://docs/svc' : null)),
  openHelp: vi.fn(),
}));
const placing = vi.hoisted(() => ({
  docked: new Set(['svc', 'loner']),
  hasDockedTile: vi.fn((/** @type {string} */ id) => placing.docked.has(id)),
  openPanelBeside: vi.fn(),
  openPanelAtTile: vi.fn(() => null),
  dockedTileOf: vi.fn(() => 'g1'),
}));
const commands = vi.hoisted(() => ({
  setPanelVisibility: vi.fn(async () => true),
  setRailOrder: vi.fn(),
}));
const groups = vi.hoisted(() => ({ moveToGroup: vi.fn() }));
const sidebarSpies = vi.hoisted(() => ({ toggleEverySession: vi.fn() }));

vi.mock(`../../../src/osprey/interfaces/web_terminal/static/js/panel-context-menu.js`, () => ({
  openContextMenu: (/** @type {any} */ opts) => captured.menus.push(opts),
}));
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/place-menu.js',
  async (original) => ({
    .../** @type {object} */ (await original()),
    registerPlace: (/** @type {any} */ place) => {
      captured.places.push(place);
      return () => {};
    },
  })
);
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
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js',
  async (original) => ({ .../** @type {object} */ (await original()), ...frames })
);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/panel-help.js',
  async (original) => ({ .../** @type {object} */ (await original()), ...help })
);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/panel-placement.js',
  async (original) => ({ .../** @type {object} */ (await original()), ...placing })
);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/panel-commands.js',
  async (original) => ({ .../** @type {object} */ (await original()), ...commands })
);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-groups.js',
  async (original) => ({ .../** @type {object} */ (await original()), ...groups })
);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/session-model.js',
  () => sessionModel
);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js', () => tree);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifact-listing.js',
  () => listing
);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js', () => confirm);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar.js', async (original) => ({
  .../** @type {object} */ (await original()),
  ...sidebarSpies,
}));

const policy = await import(`${JS}/panel-menu-policy.js`);
const { setAgentMenuRow } = await import(`${JS}/agent-menu.js`);
const { TERMINAL_RAIL_ID } = await import(`${JS}/panel-catalog.js`);
const sidebar = await import(`${JS}/sidebar.js`);
const pins = await import(`${JS}/sidebar-session-pins.js`);
const sidebarGroups = await import(`${JS}/sidebar-groups.js`);
const sidebarFold = await import(`${JS}/sidebar-fold.js`);
const treeMenus = await import(`${JS}/sidebar-artifact-menus.js`);
const verbs = await import(`${JS}/artifact-verbs.js`);
const dockMenus = await import(`${JS}/dock-menus.js`);
const { installChatMenus } = await import(`${JS}/chat-menus.js`);
const { installTerminalMenu } = await import(`${JS}/terminal-menu.js`);

/** @typedef {{label?: string, heading?: string, divider?: boolean, danger?: boolean,
 *   disabled?: boolean, submenu?: Item[], run?: () => unknown}} Item */

// ---- The rule's vocabulary (PROPOSAL.md, "Layer rule" and Functional Requirements) ----

/** Dock verbs, with the rail entry's open verbs (named in the rule's first exception). */
const DOCK_VERBS = new Set([
  'Open panel',
  'Maximize',
  'Restore',
  'Open in new window',
  'Reattach',
  'Reload',
  'Close tile',
  'Close window',
  'Close Agent pane',
  'Open in a new tile',
  'Show window',
]);
/** Exception 1: the rail entry carries its panel's open verbs in every state. */
const RAIL_ENTRY_DOCK = new Set([
  'Open in a new tile',
  'Open in new window',
  'Show window',
  'Reattach',
]);
/** Exception 2: the Sessions row carries the tileless Agent pane's rows. */
const SESSIONS_ROW_DOCK = new Set(['Open in new window', 'Reattach', 'Close Agent pane']);

/** Listing verbs: catalog, grouping, pinning. */
const LISTING_VERBS = new Set([
  'Move to',
  'Remove from rail',
  'Pin to rail',
  'New group…',
  'New group below',
  'Move up',
  'Move down',
  'Delete group',
  'Group by',
  'Pin',
  'Unpin',
  'Show pinned sessions',
  'Hide pinned sessions',
]);

/** The rows that sit last behind a divider. */
const DESTRUCTIVE = new Set([
  'Delete…',
  'Delete group',
  'Remove from rail',
  'Close tile',
  'Close window',
  'Close Agent pane',
  'Restart terminal',
  'Reset layout',
]);

/**
 * @typedef {'sidebar' | 'workspace-sidebar' | 'workspace-dock' | 'tile' | 'content'} Layer
 * @typedef {{name: string, layer: Layer, items: Item[], railEntry?: boolean,
 *   sessionsRowTileless?: boolean, offRail?: boolean}} Place
 */

/** Every label in a menu, submenus flattened. @param {Item[]} items @returns {string[]} */
function allLabels(items) {
  return items.flatMap((i) => [
    ...(i.label ? [i.label] : []),
    ...(i.submenu ? allLabels(i.submenu) : []),
  ]);
}

/** Top-level rows as the menu reads them. @param {Item[]} items */
function rows(items) {
  return items.map((i) => i.heading ?? i.label ?? (i.divider ? '—' : ''));
}

/** @param {Item[]} items @param {string} label @returns {Item} */
function row(items, label) {
  /** @param {Item[]} list @returns {Item | undefined} */
  const find = (list) => {
    for (const i of list) {
      if (i.label === label) return i;
      const inner = i.submenu && find(i.submenu);
      if (inner) return inner;
    }
    return undefined;
  };
  const hit = find(items);
  if (!hit) throw new Error(`no row "${label}" in [${rows(items).join(', ')}]`);
  return hit;
}

/**
 * The Layer rule's violations at one place.
 * @param {Place} place
 * @returns {string[]}
 */
function layerViolations(place) {
  const out = [];
  for (const label of allLabels(place.items)) {
    if (DOCK_VERBS.has(label)) {
      const ok =
        place.layer === 'tile' ||
        place.layer === 'workspace-dock' ||
        (place.railEntry && RAIL_ENTRY_DOCK.has(label)) ||
        (place.sessionsRowTileless && SESSIONS_ROW_DOCK.has(label));
      if (!ok) out.push(`dock verb "${label}" at ${place.name}`);
    }
    if (LISTING_VERBS.has(label)) {
      const ok =
        place.layer === 'sidebar' ||
        place.layer === 'workspace-sidebar' ||
        (label === 'Pin to rail' && place.layer === 'tile' && place.offRail === true);
      if (!ok) out.push(`listing verb "${label}" at ${place.name}`);
    }
    if (label === 'Reset layout' && !place.layer.startsWith('workspace')) {
      out.push(`Reset layout at ${place.name}`);
    }
  }
  return out;
}

/**
 * Whether destructive rows sit last, directly behind a divider.
 * @param {Item[]} items
 * @returns {string | null} the problem, or null
 */
function destructiveProblem(items) {
  const bad = (/** @type {Item} */ i) => Boolean(i.danger) || DESTRUCTIVE.has(i.label ?? '');
  const first = items.findIndex(bad);
  if (first < 0) return null;
  const tail = items.slice(first);
  if (!tail.every((i) => i.divider || bad(i))) {
    return `non-destructive row after a destructive one: [${rows(items).join(', ')}]`;
  }
  if (first > 0 && !items[first - 1].divider) {
    return `destructive block not behind a divider: [${rows(items).join(', ')}]`;
  }
  return null;
}

// ---- Fixtures ----

/** @type {HTMLElement} */
let rail;
/** @type {Record<string, import('vitest').Mock>} */
let deps;

/** @param {string} id @param {{failed?: boolean}} [opts] */
function railButton(id, { failed = false } = {}) {
  const li = document.createElement('li');
  li.className = 'rail-entry';
  li.dataset.id = id;
  const btn = document.createElement('button');
  btn.className = `panel-rail-button${failed ? ' failed' : ''}`;
  btn.dataset.panelId = id;
  const label = document.createElement('span');
  label.className = 'osprey-nav-label';
  label.textContent = `${id} label`;
  btn.append(label);
  li.append(btn);
  rail.append(li);
}

const members = new Set(['svc', 'closed', 'popped', 'broken', 'spare']);

beforeEach(() => {
  document.body.innerHTML = '';
  delete document.documentElement.dataset.sidebar;
  localStorage.clear();
  vi.clearAllMocks();
  captured.menus.length = 0;
  captured.places.length = 0;
  surface.terminal = true;
  dock.isTerminalPanelOpen.mockImplementation(() => true);
  dock.maximizedTileKey.mockImplementation(() => null);
  paneWindow.detached = new Set(['popped']);
  placing.docked = new Set(['svc', 'loner']);
  rail = document.createElement('ul');
  rail.id = 'panel-rail';
  for (const id of [TERMINAL_RAIL_ID, 'svc', 'closed', 'popped', 'spare']) railButton(id);
  railButton('broken', { failed: true });
  document.body.append(rail);
  deps = {
    getRailEl: vi.fn(() => rail),
    isMember: vi.fn((/** @type {string} */ id) => members.has(id)),
    getActiveTabId: vi.fn(() => null),
    activateTab: vi.fn(),
    showPanel: vi.fn(),
    retireTile: vi.fn(),
    labelOf: vi.fn((/** @type {string} */ id) => `${id} label`),
    retryStart: vi.fn(),
  };
  policy.initMenuPolicy(/** @type {any} */ (deps));
  // A panel contributing a content row, and the Agent's two extras.
  policy.setServiceMenuExtras('svc', () => [{ label: 'Svc action', run: () => {} }]);
  setAgentMenuRow('export', () => ({ label: 'Export conversation', run: exportSpy }));
  setAgentMenuRow('notifications', () => ({
    label: 'Desktop notifications',
    checked: false,
    run: notifySpy,
  }));
});

afterEach(() => {
  setAgentMenuRow('export', null);
  setAgentMenuRow('notifications', null);
  policy.setServiceMenuExtras('svc', null);
  document.body.innerHTML = '';
});

const exportSpy = vi.fn();
const notifySpy = vi.fn();

/** The Sessions row's menu in the current Agent pane state. @returns {Item[]} */
function sessionsRowItems() {
  captured.menus.length = 0;
  expect(policy.openRailContextMenu(TERMINAL_RAIL_ID, 0, 0)).toBe(true);
  return captured.menus.at(-1).items;
}

/** A group heading's menu. @param {string} id @returns {Item[]} */
function groupHeadingItems(id) {
  captured.menus.length = 0;
  expect(sidebarGroups.openGroupMenu(rail, id, 0, 0)).toBe(true);
  return captured.menus.at(-1).items;
}

/** The session-tree DOM the row builders read. */
function sessionTree() {
  const box = document.createElement('div');
  box.dataset.sessionTree = '';
  box.innerHTML = `
    <div class="osprey-tree-session osprey-tree-d0" data-session-id="t1" data-title-source="user"
         data-pinned="1" aria-expanded="true"><span class="osprey-tree-title">Beam study</span></div>
    <div class="osprey-tree-session osprey-tree-d0 active" data-session-id="t2"
         aria-expanded="true"><span class="osprey-tree-title">Now</span></div>
    <div class="osprey-tree-group" data-group-head="t1:plots" aria-expanded="true">Plots</div>
    <div class="osprey-tree-group" data-group-head="t1:tables" aria-expanded="true">Tables</div>
    <div class="osprey-list-row osprey-tree-run" data-id="run1" data-run="run1">orbit scan</div>
    <div class="osprey-list-row" data-id="a1">BPM plot</div>`;
  document.body.append(box);
  const q = (/** @type {string} */ s) => /** @type {HTMLElement} */ (box.querySelector(s));
  return {
    session: q('[data-session-id="t1"]'),
    active: q('[data-session-id="t2"]'),
    group: q('[data-group-head="t1:plots"]'),
    run: q('[data-run="run1"]'),
    artifact: q('[data-id="a1"]:not([data-run])'),
  };
}

const RUN = {
  id: 'run1',
  exec_id: 'exec-7',
  title: 'orbit scan',
  run_parts: [
    { id: 'run1', role: 'head' },
    { id: 'out1', role: 'result' },
    { id: 'pre1', role: 'preview' },
    { id: 'code1', role: 'code' },
  ],
};
const PLOT = { id: 'a1', title: 'BPM plot', filename: 'bpm.png' };

/** The pinned heading (made by the group's first paint). @returns {HTMLElement} */
function pinnedHead() {
  const head = document.createElement('div');
  head.className = 'sidebar-pinned-head';
  const name = document.createElement('span');
  name.className = 'sidebar-group-name';
  name.textContent = 'Pinned';
  head.append(name);
  document.body.append(head);
  return head;
}

/** The chat's three places and the terminal body's, captured from registerPlace. */
function contentPlaces() {
  captured.places.length = 0;
  const messages = document.createElement('div');
  messages.className = 'messages';
  const entry = document.createElement('div');
  entry.className = 'chat-entry operator';
  entry.textContent = 'hello';
  const runRow = document.createElement('div');
  runRow.className = 'op-row op-tool';
  runRow.dataset.tool = 'execute';
  runRow.dataset.callId = 'toolu_7';
  messages.append(entry, runRow);
  const textarea = document.createElement('textarea');
  document.body.append(messages, textarea);
  installChatMenus({
    messages,
    textarea,
    scrollToBottom: () => {},
    onNewSession: () => {},
    onExport: () => {},
  });
  const container = document.createElement('div');
  document.body.append(container);
  installTerminalMenu({
    container,
    term: {
      hasSelection: () => true,
      getSelection: () => 'x',
      selectAll: () => {},
      clear: () => {},
      paste: () => {},
    },
    onRestart: () => {},
    onSwitchToChat: () => {},
  });
  const [entryPlace, runPlace, logPlace, termPlace] = captured.places;
  const ev = (/** @type {Element} */ target) => ({ target, clientX: 0, clientY: 0 });
  return {
    message: entryPlace.build(entry, ev(entry)),
    run: runPlace.build(runRow, ev(runRow)),
    background: logPlace.build(messages, ev(messages)),
    terminal: termPlace.build(container, ev(container)),
  };
}

/**
 * Every shell place in its representative states.
 * @returns {Place[]}
 */
function everyPlace() {
  /** @type {Place[]} */
  const out = [];
  // ---- Sidebar places ----
  out.push({
    name: 'rail entry (docked)',
    layer: 'sidebar',
    railEntry: true,
    items: policy.buildServiceMenuItems('svc'),
  });
  out.push({
    name: 'rail entry (tile closed)',
    layer: 'sidebar',
    railEntry: true,
    items: policy.buildServiceMenuItems('closed'),
  });
  out.push({
    name: 'rail entry (detached)',
    layer: 'sidebar',
    railEntry: true,
    items: policy.buildServiceMenuItems('popped'),
  });
  out.push({
    name: 'rail entry (failed)',
    layer: 'sidebar',
    railEntry: true,
    items: policy.buildServiceMenuItems('broken'),
  });
  sidebarGroups.addGroup('Custom');
  const custom = sidebarGroups
    .loadGroups()
    .groups.find((/** @type {any} */ g) => g.label === 'Custom');
  out.push({
    name: 'group heading (built-in)',
    layer: 'sidebar',
    items: groupHeadingItems('tools'),
  });
  out.push({
    name: 'group heading (custom)',
    layer: 'sidebar',
    items: groupHeadingItems(/** @type {any} */ (custom).id),
  });
  dock.isTerminalPanelOpen.mockImplementation(() => true);
  out.push({ name: 'Sessions row (Agent tiled)', layer: 'sidebar', items: sessionsRowItems() });
  dock.isTerminalPanelOpen.mockImplementation(() => false);
  out.push({
    name: 'Sessions row (Agent tileless)',
    layer: 'sidebar',
    sessionsRowTileless: true,
    items: sessionsRowItems(),
  });
  paneWindow.detached.add(TERMINAL_RAIL_ID);
  out.push({
    name: 'Sessions row (Agent detached)',
    layer: 'sidebar',
    sessionsRowTileless: true,
    items: sessionsRowItems(),
  });
  paneWindow.detached.delete(TERMINAL_RAIL_ID);
  dock.isTerminalPanelOpen.mockImplementation(() => true);
  const t = sessionTree();
  out.push({ name: 'session row', layer: 'sidebar', items: treeMenus.sessionMenuItems(t.session) });
  out.push({
    name: 'session row (active)',
    layer: 'sidebar',
    items: treeMenus.sessionMenuItems(t.active),
  });
  out.push({
    name: 'run row',
    layer: 'sidebar',
    items: verbs.runMenuItems(RUN.exec_id || RUN.id, RUN),
  });
  out.push({
    name: 'artifact row',
    layer: 'sidebar',
    items: verbs.artifactMenuItems(PLOT, { open: () => t.artifact.click() }),
  });
  out.push({ name: 'tree group', layer: 'sidebar', items: treeMenus.groupMenuItems(t.group) });
  out.push({ name: 'pinned heading', layer: 'sidebar', items: pins.headMenuItems(pinnedHead()) });
  // ---- Workspace places ----
  out.push({
    name: 'sidebar header',
    layer: 'workspace-sidebar',
    items: sidebar.headerMenuItems(),
  });
  document.documentElement.dataset.sidebar = 'collapsed';
  out.push({
    name: 'sidebar header (collapsed)',
    layer: 'workspace-sidebar',
    items: sidebar.headerMenuItems(),
  });
  delete document.documentElement.dataset.sidebar;
  out.push({
    name: 'empty sidebar',
    layer: 'workspace-sidebar',
    items: pins.emptySidebarMenuItems(),
  });
  out.push({
    name: 'empty dock slot',
    layer: 'workspace-dock',
    items: dockMenus.buildEmptyDockItems(document),
  });
  // ---- Tile places ----
  out.push({ name: 'service tile bar', layer: 'tile', items: policy.buildTileMenuItems('svc') });
  out.push({
    name: 'service tile bar (off-rail)',
    layer: 'tile',
    offRail: true,
    items: policy.buildTileMenuItems('loner'),
  });
  out.push({
    name: 'service tile bar (fallback)',
    layer: 'tile',
    items: policy.buildTileMenuItems('svc', { mode: 'fallback' }),
  });
  out.push({
    name: 'pop-out bar',
    layer: 'tile',
    items: policy.buildTileMenuItems('popped', { mode: 'detached' }),
  });
  paneWindow.detached.add('loner');
  out.push({
    name: 'pop-out bar (off-rail)',
    layer: 'tile',
    offRail: true,
    items: policy.buildTileMenuItems('loner', { mode: 'detached' }),
  });
  paneWindow.detached.delete('loner');
  dock.maximizedTileKey.mockImplementation(() => 'svc');
  out.push({
    name: 'service tile bar (maximized)',
    layer: 'tile',
    items: policy.buildTileMenuItems('svc'),
  });
  dock.maximizedTileKey.mockImplementation(() => null);
  surface.terminal = true;
  out.push({
    name: 'Agent tile bar (terminal)',
    layer: 'tile',
    items: policy.buildTerminalMenuItems({}),
  });
  out.push({
    name: 'Agent tile bar (fallback)',
    layer: 'tile',
    items: policy.buildTerminalMenuItems({ mode: 'fallback' }),
  });
  surface.terminal = false;
  out.push({
    name: 'Agent tile bar (chat)',
    layer: 'tile',
    items: policy.buildTerminalMenuItems({}),
  });
  paneWindow.detached.add(TERMINAL_RAIL_ID);
  out.push({
    name: 'Agent pop-out bar',
    layer: 'tile',
    items: policy.buildTerminalMenuItems({ mode: 'detached' }),
  });
  paneWindow.detached.delete(TERMINAL_RAIL_ID);
  surface.terminal = true;
  // ---- Content places ----
  const content = contentPlaces();
  out.push({ name: 'chat message', layer: 'content', items: content.message });
  out.push({ name: 'chat run row', layer: 'content', items: content.run });
  out.push({ name: 'chat background', layer: 'content', items: content.background });
  out.push({ name: 'terminal body', layer: 'content', items: content.terminal });
  return out;
}

// ---- 1. The Layer rule ----

describe('the Layer rule holds at every shell place', () => {
  test('every place yields rows in its representative states', () => {
    for (const place of everyPlace()) {
      expect(place.items.length, place.name).toBeGreaterThan(0);
    }
  });

  test('no dock verb, listing verb or Reset layout sits outside its layer or named exception', () => {
    const violations = everyPlace().flatMap(layerViolations);
    expect(violations).toEqual([]);
  });

  test('the Sessions row carries no Agent pane rows while the pane has a tile', () => {
    dock.isTerminalPanelOpen.mockImplementation(() => true);
    const labels = allLabels(sessionsRowItems());
    for (const verb of SESSIONS_ROW_DOCK) expect(labels).not.toContain(verb);
  });

  test('Pin to rail reaches a tile bar only for an off-rail panel', () => {
    expect(allLabels(policy.buildTileMenuItems('svc'))).not.toContain('Pin to rail');
    expect(allLabels(policy.buildTileMenuItems('loner'))).toContain('Pin to rail');
  });

  test('the checker itself rejects a dock verb at a Sidebar place and a listing verb at a Tile place', () => {
    expect(
      layerViolations({ name: 'x', layer: 'sidebar', items: [{ label: 'Close tile' }] })
    ).toHaveLength(1);
    expect(
      layerViolations({ name: 'x', layer: 'tile', items: [{ label: 'Move to', submenu: [] }] })
    ).toHaveLength(1);
    expect(
      layerViolations({ name: 'x', layer: 'tile', items: [{ label: 'Reset layout' }] })
    ).toHaveLength(1);
    expect(
      layerViolations({ name: 'x', layer: 'content', items: [{ label: 'Maximize' }] })
    ).toHaveLength(1);
  });
});

// ---- 3. Destructive rows last ----

/** Places whose destructive-last defect is pinned by a test.fails below. */
const KNOWN_GAPS = new Set(['Sessions row (Agent detached)']);

describe('destructive rows sit last behind a divider', () => {
  test('in every shell menu', () => {
    const problems = everyPlace()
      .filter((p) => !KNOWN_GAPS.has(p.name))
      .map((p) => [p.name, destructiveProblem(p.items)])
      .filter(([, problem]) => problem !== null);
    expect(problems).toEqual([]);
  });

  // Known gap, reported for panel-menu-policy.js (tilelessAgentRows): with the
  // Agent pane detached, the Sessions row ends "—, Reattach, Close Agent pane",
  // so Close Agent pane is last but not directly behind a divider. Expected to
  // fail until fixed; once it passes, vitest reports this test and the entry
  // in KNOWN_GAPS comes out.
  test.fails('in the Sessions row with the Agent pane detached (known gap)', () => {
    const place = everyPlace().find((p) => p.name === 'Sessions row (Agent detached)');
    expect(destructiveProblem(/** @type {Place} */ (place).items)).toBeNull();
  });

  test('the checker rejects a destructive row before a safe one and one without a divider', () => {
    expect(destructiveProblem([{ label: 'Delete…' }, { label: 'Copy ID' }])).not.toBeNull();
    expect(destructiveProblem([{ label: 'Copy ID' }, { label: 'Delete…' }])).not.toBeNull();
    expect(
      destructiveProblem([{ label: 'Copy ID' }, { divider: true }, { label: 'Delete…' }])
    ).toBeNull();
  });
});

// ---- 2. Every pre-existing verb still reachable, still on its old handler ----

describe('every GP7 verb is reachable at its place and calls its previous handler', () => {
  test('rail entry: Focus (now Focus tile) activates the panel', () => {
    row(policy.buildServiceMenuItems('svc'), 'Focus tile').run?.();
    expect(deps.activateTab).toHaveBeenCalledExactlyOnceWith('svc', { userInitiated: true });
  });

  test('rail entry: Open in a new tile opens beside', () => {
    row(policy.buildServiceMenuItems('svc'), 'Open in a new tile').run?.();
    expect(placing.openPanelBeside).toHaveBeenCalledExactlyOnceWith('svc');
  });

  test('rail entry: Open in new window detaches and Reattach reattaches', () => {
    row(policy.buildServiceMenuItems('svc'), 'Open in new window').run?.();
    expect(paneWindow.detach).toHaveBeenCalledExactlyOnceWith('svc');
    row(policy.buildServiceMenuItems('popped'), 'Reattach').run?.();
    expect(paneWindow.reattach).toHaveBeenCalledOnce();
    expect(paneWindow.reattach.mock.calls[0][0]).toBe('popped');
  });

  test('rail entry: Move to <group> moves, and Move to new group… (now New group…) creates and moves', () => {
    const items = policy.buildServiceMenuItems('svc');
    const move = row(items, 'Move to');
    const target = /** @type {Item[]} */ (move.submenu).find(
      (i) => i.label && i.label !== 'New group…'
    );
    target?.run?.();
    expect(groups.moveToGroup).toHaveBeenCalledOnce();
    expect(groups.moveToGroup.mock.calls[0][0]).toBe('svc');
    const before = sidebarGroups.loadGroups().groups.length;
    row(move.submenu ?? [], 'New group…').run?.();
    expect(sidebarGroups.loadGroups().groups.length).toBe(before + 1);
    expect(groups.moveToGroup).toHaveBeenCalledTimes(2);
  });

  test('rail entry: Remove from rail clears membership; Pin to rail (off-rail) sets it', async () => {
    row(policy.buildServiceMenuItems('svc'), 'Remove from rail').run?.();
    await vi.waitFor(() => expect(commands.setPanelVisibility).toHaveBeenCalledWith('svc', false));
    row(policy.buildTileMenuItems('loner'), 'Pin to rail').run?.();
    expect(commands.setPanelVisibility).toHaveBeenCalledWith('loner', true);
  });

  test('service tile bar: Reload, Help and Open in new window keep their handlers', () => {
    const items = policy.buildTileMenuItems('svc');
    row(items, 'Reload').run?.();
    expect(frames.reloadPanel).toHaveBeenCalledExactlyOnceWith('svc');
    row(items, 'Help').run?.();
    expect(help.openHelp).toHaveBeenCalledExactlyOnceWith('svc');
    row(items, 'Open in new window').run?.();
    expect(paneWindow.detach).toHaveBeenCalledExactlyOnceWith('svc');
  });

  test('pop-out bar: Reattach and Reload keep their handlers', () => {
    const items = policy.buildTileMenuItems('popped', { mode: 'detached' });
    row(items, 'Reattach').run?.();
    expect(paneWindow.reattach.mock.calls[0][0]).toBe('popped');
    row(items, 'Reload').run?.();
    expect(frames.reloadPanel).toHaveBeenCalledExactlyOnceWith('popped');
  });

  test('Reset layout (left the tile bars) resets the dock from the sidebar header and the empty dock slot', async () => {
    row(sidebar.headerMenuItems(), 'Reset layout').run?.();
    await vi.waitFor(() => expect(dock.resetDockLayout).toHaveBeenCalledTimes(1));
    row(dockMenus.buildEmptyDockItems(document), 'Reset layout').run?.();
    await vi.waitFor(() => expect(dock.resetDockLayout).toHaveBeenCalledTimes(2));
  });

  test('Agent tile bar: Open in new window, Close Agent pane and Restart terminal keep their handlers', async () => {
    surface.terminal = true;
    const items = policy.buildTerminalMenuItems({});
    row(items, 'Open in new window').run?.();
    expect(paneWindow.detach).toHaveBeenCalledExactlyOnceWith(TERMINAL_RAIL_ID);
    row(items, 'Close Agent pane').run?.();
    expect(dock.closeTerminalPanel).toHaveBeenCalledOnce();
    await row(items, 'Restart terminal').run?.();
    expect(term.restartTerminal).toHaveBeenCalledOnce();
    expect(term.startTerminal).toHaveBeenCalledOnce();
  });

  test('Agent tile bar: Export conversation and Desktop notifications run their rows', () => {
    const items = policy.buildTerminalMenuItems({});
    row(items, 'Export conversation').run?.();
    row(items, 'Desktop notifications').run?.();
    expect(exportSpy).toHaveBeenCalledOnce();
    expect(notifySpy).toHaveBeenCalledOnce();
  });

  test('Sessions row: New session, Group by Kind / Activity and Collapse all / Expand all', async () => {
    const items = sessionsRowItems();
    row(items, 'New session').run?.();
    await vi.waitFor(() =>
      expect(sessionModel.switchTo).toHaveBeenCalledExactlyOnceWith({ new: true })
    );
    row(items, 'Activity').run?.();
    await vi.waitFor(() => expect(tree.setTreeView).toHaveBeenCalledWith('activity'));
    row(items, 'Kind').run?.();
    await vi.waitFor(() => expect(tree.setTreeView).toHaveBeenCalledWith('kind'));
    const all = items.find((i) => i.label === 'Collapse all' || i.label === 'Expand all');
    all?.run?.();
    await vi.waitFor(() => expect(sidebarSpies.toggleEverySession).toHaveBeenCalledOnce());
  });

  test('Sessions row (tileless): Open in new window, Reattach and Close Agent pane', () => {
    dock.isTerminalPanelOpen.mockImplementation(() => false);
    row(sessionsRowItems(), 'Open in new window').run?.();
    expect(paneWindow.detach).toHaveBeenCalledExactlyOnceWith(TERMINAL_RAIL_ID);
    paneWindow.detached.add(TERMINAL_RAIL_ID);
    const detached = sessionsRowItems();
    row(detached, 'Reattach').run?.();
    expect(paneWindow.reattach.mock.calls[0][0]).toBe(TERMINAL_RAIL_ID);
    row(detached, 'Close Agent pane').run?.();
    expect(dock.closeTerminalPanel).toHaveBeenCalledOnce();
  });

  test('session row: Open, Pin / Unpin, Rename…, Use automatic name, Export transcript, Copy ID, Delete…', async () => {
    const fetchSpy = vi.fn(async () => ({ ok: true, json: async () => ({}) }));
    vi.stubGlobal('fetch', fetchSpy);
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    try {
      const t = sessionTree();
      const click = vi.fn();
      t.session.addEventListener('click', click);
      const items = treeMenus.sessionMenuItems(t.session);
      row(items, 'Open').run?.();
      expect(click).toHaveBeenCalledOnce();
      row(items, 'Unpin').run?.();
      await vi.waitFor(() =>
        expect(sessionModel.setPinned).toHaveBeenCalledExactlyOnceWith('t1', false)
      );
      row(items, 'Use automatic name').run?.();
      await vi.waitFor(() => expect(fetchSpy).toHaveBeenCalledOnce());
      expect(/** @type {any} */ (fetchSpy.mock.calls[0])[1].method).toBe('PUT');
      row(items, 'Export transcript').run?.();
      expect(anchorClick).toHaveBeenCalledOnce();
      row(items, 'Copy ID').run?.();
      expect(writeText).toHaveBeenCalledExactlyOnceWith('t1');
      row(items, 'Delete…').run?.();
      expect(confirm.showConfirm).toHaveBeenCalledOnce();
      row(items, 'Rename…').run?.();
      expect(t.session.querySelector('input.osprey-tree-rename')).not.toBeNull();
      // An unpinned session offers Pin, through the same handler.
      delete t.session.dataset.pinned;
      row(treeMenus.sessionMenuItems(t.session), 'Pin').run?.();
      await vi.waitFor(() => expect(sessionModel.setPinned).toHaveBeenCalledWith('t1', true));
    } finally {
      anchorClick.mockRestore();
      vi.unstubAllGlobals();
    }
  });

  test('tree group: Collapse / Expand clicks the heading', () => {
    const t = sessionTree();
    const click = vi.fn();
    t.group.addEventListener('click', click);
    const toggle = treeMenus
      .groupMenuItems(t.group)
      .find((/** @type {any} */ i) => i.label === 'Collapse' || i.label === 'Expand');
    toggle?.run?.();
    expect(click).toHaveBeenCalledOnce();
  });

  test('run and artifact rows: Open, Copy ID, Delete… keep their handlers', async () => {
    const writeText = vi.fn(async () => {});
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    const t = sessionTree();
    const click = vi.fn();
    t.artifact.addEventListener('click', click);
    const art = verbs.artifactMenuItems(PLOT, { open: () => t.artifact.click() });
    row(art, 'Open').run?.();
    expect(click).toHaveBeenCalledOnce();
    row(art, 'Copy ID').run?.();
    expect(writeText).toHaveBeenCalledWith('a1');
    row(art, 'Delete…').run?.();
    expect(confirm.showConfirm).toHaveBeenCalledTimes(1);
    row(verbs.runMenuItems(RUN.exec_id || RUN.id, RUN), 'Delete…').run?.();
    await vi.waitFor(() => expect(confirm.showConfirm).toHaveBeenCalledTimes(2));
  });

  test('group heading: Rename (now Rename…), New group below, Move up / down, Collapse, Delete group', () => {
    sidebarGroups.addGroup('Custom');
    const id = /** @type {any} */ (
      sidebarGroups.loadGroups().groups.find((/** @type {any} */ g) => g.label === 'Custom')
    ).id;
    const ids = () => sidebarGroups.loadGroups().groups.map((/** @type {any} */ g) => g.id);
    // Rename… swaps the name for a field, as Rename did.
    const groupEl = document.createElement('div');
    groupEl.className = 'sidebar-group';
    groupEl.dataset.group = id;
    const name = document.createElement('span');
    name.className = 'sidebar-group-name';
    name.textContent = 'Custom';
    groupEl.append(name);
    rail.append(groupEl);
    row(groupHeadingItems(id), 'Rename…').run?.();
    expect(groupEl.querySelector('.sidebar-group-rename')).not.toBeNull();
    groupEl.remove();

    const count = ids().length;
    row(groupHeadingItems(id), 'New group below').run?.();
    expect(ids().length).toBe(count + 1);
    expect(ids()[ids().indexOf(id) + 1]).not.toBe(undefined);

    const at = ids().indexOf(id);
    row(groupHeadingItems(id), 'Move up').run?.();
    expect(ids().indexOf(id)).toBe(at - 1);
    row(groupHeadingItems(id), 'Move down').run?.();
    expect(ids().indexOf(id)).toBe(at);

    // Collapse / Expand group fold the group (sidebar-fold.js, wired to this rail).
    sidebarFold.initSidebarFold(rail);
    row(groupHeadingItems(id), 'Collapse group').run?.();
    expect(sidebarFold.isGroupFolded(id)).toBe(true);
    row(groupHeadingItems(id), 'Expand group').run?.();
    expect(sidebarFold.isGroupFolded(id)).toBe(false);

    row(groupHeadingItems(id), 'Delete group').run?.();
    expect(ids()).not.toContain(id);
  });

  test('pinned heading: Collapse / Expand, Rename (now Rename…), Hide pinned sessions', () => {
    const head = pinnedHead();
    pins.paintPinnedGroup(1);
    const first = pins
      .headMenuItems(head)
      .find((/** @type {any} */ i) => i.label === 'Collapse' || i.label === 'Expand');
    const flipped = first?.label === 'Collapse' ? 'Expand' : 'Collapse';
    first?.run?.();
    expect(rows(pins.headMenuItems(head))).toContain(flipped);
    row(pins.headMenuItems(head), 'Rename…').run?.();
    expect(head.querySelector('input.sidebar-group-rename')).not.toBeNull();
    expect(pins.pinnedShown()).toBe(true);
    row(pins.headMenuItems(head), 'Hide pinned sessions').run?.();
    expect(pins.pinnedShown()).toBe(false);
    // Show pinned sessions (Sessions row, empty sidebar) brings it back.
    row(pins.emptySidebarMenuItems(), 'Show pinned sessions').run?.();
    expect(pins.pinnedShown()).toBe(true);
    pins.paintPinnedGroup(0);
  });
});
