/**
 * Every menu hint names a real key binding, on both platforms.
 *
 * Every shell place's rows are built in the representative states
 * menu-layers.test.mjs uses (Sidebar, Workspace, Tile and Content places),
 * once with a macOS platform and once with a non-mac one, and each level —
 * submenus included — is rendered through the design system's context menu.
 * Every rendered hint cell is collected and must:
 *
 *   1. equal `hintFor(id)` of an entry of key-bindings.js under that platform,
 *   2. be backed on that platform: the entry's descriptor either has a matcher
 *      that accepts the chord the hint spells, or is flagged native (the
 *      browser or xterm performs it, e.g. ⌘C copying a terminal selection).
 *
 * Paste, Clear terminal and Select all carry no hint: they have no binding.
 *
 *   npx vitest run tests/interfaces/web_terminal/menu-hints.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

import { closeContextMenu, openContextMenu as renderMenu } from '/design-system/js/context-menu.js';
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
const treeMenus = await import(`${JS}/sidebar-artifact-menus.js`);
const dockMenus = await import(`${JS}/dock-menus.js`);
const { installChatMenus } = await import(`${JS}/chat-menus.js`);
const { installTerminalMenu } = await import(`${JS}/terminal-menu.js`);

const { KEY_BINDINGS, bindingFor, hintFor } = await import(`${JS}/key-bindings.js`);

/** @typedef {{label?: string, heading?: string, divider?: boolean, hint?: string,
 *   submenu?: Item[], run?: () => unknown}} Item */
/** @typedef {{name: string, items: Item[]}} Place */

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

/** The chat's two places and the terminal body's, captured from registerPlace. */
function contentPlaces() {
  captured.places.length = 0;
  const messages = document.createElement('div');
  messages.className = 'messages';
  const entry = document.createElement('div');
  entry.className = 'chat-entry operator';
  entry.textContent = 'hello';
  messages.append(entry);
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
  const [entryPlace, logPlace, termPlace] = captured.places;
  const ev = (/** @type {Element} */ target) => ({ target, clientX: 0, clientY: 0 });
  return {
    message: entryPlace.build(entry, ev(entry)),
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
    items: policy.buildServiceMenuItems('svc'),
  });
  out.push({
    name: 'rail entry (tile closed)',
    items: policy.buildServiceMenuItems('closed'),
  });
  out.push({
    name: 'rail entry (detached)',
    items: policy.buildServiceMenuItems('popped'),
  });
  out.push({
    name: 'rail entry (failed)',
    items: policy.buildServiceMenuItems('broken'),
  });
  sidebarGroups.addGroup('Custom');
  const custom = sidebarGroups
    .loadGroups()
    .groups.find((/** @type {any} */ g) => g.label === 'Custom');
  out.push({
    name: 'group heading (built-in)',
    items: groupHeadingItems('tools'),
  });
  out.push({
    name: 'group heading (custom)',
    items: groupHeadingItems(/** @type {any} */ (custom).id),
  });
  dock.isTerminalPanelOpen.mockImplementation(() => true);
  out.push({ name: 'Sessions row (Agent tiled)', items: sessionsRowItems() });
  dock.isTerminalPanelOpen.mockImplementation(() => false);
  out.push({
    name: 'Sessions row (Agent tileless)',
    items: sessionsRowItems(),
  });
  paneWindow.detached.add(TERMINAL_RAIL_ID);
  out.push({
    name: 'Sessions row (Agent detached)',
    items: sessionsRowItems(),
  });
  paneWindow.detached.delete(TERMINAL_RAIL_ID);
  dock.isTerminalPanelOpen.mockImplementation(() => true);
  const t = sessionTree();
  out.push({ name: 'session row', items: treeMenus.sessionMenuItems(t.session) });
  out.push({
    name: 'session row (active)',
    items: treeMenus.sessionMenuItems(t.active),
  });
  out.push({ name: 'run row', items: treeMenus.runMenuItems(t.run, RUN) });
  out.push({
    name: 'artifact row',
    items: treeMenus.artifactMenuItems(t.artifact, PLOT),
  });
  out.push({ name: 'tree group', items: treeMenus.groupMenuItems(t.group) });
  out.push({ name: 'pinned heading', items: pins.headMenuItems(pinnedHead()) });
  // ---- Workspace places ----
  out.push({
    name: 'sidebar header',
    items: sidebar.headerMenuItems(),
  });
  document.documentElement.dataset.sidebar = 'collapsed';
  out.push({
    name: 'sidebar header (collapsed)',
    items: sidebar.headerMenuItems(),
  });
  delete document.documentElement.dataset.sidebar;
  out.push({
    name: 'empty sidebar',
    items: pins.emptySidebarMenuItems(),
  });
  out.push({
    name: 'empty dock slot',
    items: dockMenus.buildEmptyDockItems(document),
  });
  // ---- Tile places ----
  out.push({ name: 'service tile bar', items: policy.buildTileMenuItems('svc') });
  out.push({
    name: 'service tile bar (off-rail)',
    items: policy.buildTileMenuItems('loner'),
  });
  out.push({
    name: 'service tile bar (fallback)',
    items: policy.buildTileMenuItems('svc', { mode: 'fallback' }),
  });
  out.push({
    name: 'pop-out bar',
    items: policy.buildTileMenuItems('popped', { mode: 'detached' }),
  });
  paneWindow.detached.add('loner');
  out.push({
    name: 'pop-out bar (off-rail)',
    items: policy.buildTileMenuItems('loner', { mode: 'detached' }),
  });
  paneWindow.detached.delete('loner');
  dock.maximizedTileKey.mockImplementation(() => 'svc');
  out.push({
    name: 'service tile bar (maximized)',
    items: policy.buildTileMenuItems('svc'),
  });
  dock.maximizedTileKey.mockImplementation(() => null);
  surface.terminal = true;
  out.push({
    name: 'Agent tile bar (terminal)',
    items: policy.buildTerminalMenuItems({}),
  });
  out.push({
    name: 'Agent tile bar (fallback)',
    items: policy.buildTerminalMenuItems({ mode: 'fallback' }),
  });
  surface.terminal = false;
  out.push({
    name: 'Agent tile bar (chat)',
    items: policy.buildTerminalMenuItems({}),
  });
  paneWindow.detached.add(TERMINAL_RAIL_ID);
  out.push({
    name: 'Agent pop-out bar',
    items: policy.buildTerminalMenuItems({ mode: 'detached' }),
  });
  paneWindow.detached.delete(TERMINAL_RAIL_ID);
  surface.terminal = true;
  // ---- Content places ----
  const content = contentPlaces();
  out.push({ name: 'chat message', items: content.message });
  out.push({ name: 'chat background', items: content.background });
  out.push({ name: 'terminal body', items: content.terminal });
  return out;
}

// ---- Platform ----

/** @param {string} value */
function stubPlatform(value) {
  Object.defineProperty(navigator, 'platform', { value, configurable: true });
  Object.defineProperty(navigator, 'userAgentData', { value: undefined, configurable: true });
}

afterEach(() => {
  closeContextMenu();
  stubPlatform('');
});

const PLATFORMS = [
  { name: 'mac', platform: 'MacIntel', mac: true },
  { name: 'other', platform: 'Linux x86_64', mac: false },
];

// ---- Hint cells, as the design system renders them ----

/** @typedef {{place: string, label: string, hint: string}} HintCell */

/**
 * Render a menu level — and each submenu as its own level — through the
 * design system's context menu, and read back every hint cell.
 * @param {string} place
 * @param {Item[]} items
 * @returns {HintCell[]}
 */
function renderedHints(place, items) {
  renderMenu({ x: 0, y: 0, ariaLabel: place, items: /** @type {any} */ (items) });
  const menu = /** @type {HTMLElement} */ (document.body.querySelector('.osprey-context-menu'));
  const cells = [...menu.querySelectorAll('.osprey-context-item')].flatMap((btn) => {
    const hint = btn.querySelector('.osprey-context-hint');
    if (!hint) return [];
    const label = btn.querySelector('.osprey-context-label')?.textContent ?? '';
    return [{ place, label, hint: hint.textContent ?? '' }];
  });
  closeContextMenu();
  const nested = items.flatMap((i) =>
    i.submenu ? renderedHints(`${place} › ${i.label}`, i.submenu) : []
  );
  return [...cells, ...nested];
}

/** Every hint cell of every shell place, under the current platform. @returns {HintCell[]} */
function everyHint() {
  return everyPlace().flatMap((p) => renderedHints(p.name, p.items));
}

// ---- The checker ----

/**
 * The key event a hint spells: ⌘ / Cmd+ is Meta, ⌃ / Ctrl+ is Control,
 * ⇧ / Shift+ is Shift, ⌥ / Alt+ is Alt; what remains is `event.key`.
 * @param {string} hint
 * @returns {KeyboardEvent}
 */
function chordEvent(hint) {
  const init = { metaKey: false, ctrlKey: false, shiftKey: false, altKey: false };
  let rest = hint;
  const mods = /** @type {[RegExp, keyof typeof init][]} */ ([
    [/^(⌘|Cmd\+)/, 'metaKey'],
    [/^(⌃|Ctrl\+)/, 'ctrlKey'],
    [/^(⇧|Shift\+)/, 'shiftKey'],
    [/^(⌥|Alt\+|Option\+)/, 'altKey'],
  ]);
  for (let hit = true; hit;) {
    hit = false;
    for (const [re, mod] of mods) {
      const m = rest.match(re);
      if (m && rest.length > m[0].length) {
        init[mod] = true;
        rest = rest.slice(m[0].length);
        hit = true;
      }
    }
  }
  return new KeyboardEvent('keydown', { key: rest, bubbles: true, cancelable: true, ...init });
}

/**
 * What is wrong with one hint cell on a platform, or null.
 * @param {HintCell} cell
 * @param {boolean} mac
 * @param {Record<string, any>} [bindings]
 * @returns {string | null}
 */
function hintProblem(cell, mac, bindings = KEY_BINDINGS) {
  const where = `"${cell.hint}" on ${cell.place} › ${cell.label}`;
  const side = mac ? 'mac' : 'other';
  const ids = Object.keys(bindings).filter((id) => bindings[id][side].hint === cell.hint);
  if (ids.length === 0) return `${where} is no key-bindings.js hint`;
  const backed = ids.some((id) => {
    const d = bindings[id][side];
    return (
      d.native === true || (typeof d.matcher === 'function' && d.matcher(chordEvent(cell.hint)))
    );
  });
  return backed
    ? null
    : `${where} (${ids.join(', ')}) is neither native nor accepted by its matcher`;
}

// ---- Tests ----

/** The rows carrying a hint today, per place. */
const HINTED_TODAY = [
  ['sidebar header', 'Collapse sidebar', 'sidebar-toggle'],
  ['sidebar header (collapsed)', 'Expand sidebar', 'sidebar-toggle'],
  ['empty sidebar', 'Collapse sidebar', 'sidebar-toggle'],
  ['terminal body', 'Copy', 'terminal-copy'],
];

describe.each(PLATFORMS)('menu hints on $name', ({ platform, mac }) => {
  beforeEach(() => stubPlatform(platform));

  test('the stub selects the platform hintFor reads', () => {
    expect(hintFor('terminal-copy')).toBe(bindingFor('terminal-copy', mac).hint);
  });

  test('the hinted rows are the ones expected today, each with its binding’s hint', () => {
    const cells = everyHint().map((c) => [c.place, c.label, c.hint]);
    const expected = HINTED_TODAY.map(([place, label, id]) => [place, label, hintFor(id)]);
    expect(cells.sort()).toEqual(expected.sort());
  });

  test('every hint cell equals hintFor of a binding that is native or whose matcher accepts it', () => {
    const cells = everyHint();
    expect(cells.length).toBeGreaterThan(0);
    const problems = cells.map((c) => hintProblem(c, mac)).filter((p) => p !== null);
    expect(problems).toEqual([]);
    // What the row prints is what hintFor returns under this platform.
    const printed = new Set(Object.keys(KEY_BINDINGS).map((id) => hintFor(id)));
    for (const c of cells) expect(printed.has(c.hint), c.hint).toBe(true);
  });

  test('Paste, Clear terminal and Select all carry no hint', () => {
    const terminal = everyPlace().find((p) => p.name === 'terminal body');
    const items = /** @type {Place} */ (terminal).items;
    for (const label of ['Paste', 'Clear terminal', 'Select all']) {
      const item = items.find((i) => i.label === label);
      expect(item, label).toBeDefined();
      expect(item?.hint, label).toBeUndefined();
    }
    expect(renderedHints('terminal body', items).map((c) => c.label)).toEqual(['Copy']);
  });
});

describe('the checker', () => {
  test('parses a hint into the chord it spells', () => {
    const e = chordEvent('Ctrl+Shift+C');
    expect([e.type, e.ctrlKey, e.shiftKey, e.metaKey, e.key]).toEqual([
      'keydown',
      true,
      true,
      false,
      'C',
    ]);
    const m = chordEvent('⌘\\');
    expect([m.metaKey, m.ctrlKey, m.key]).toEqual([true, false, '\\']);
  });

  test('rejects a hint no binding prints', () => {
    stubPlatform('Linux x86_64');
    expect(hintProblem({ place: 'x', label: 'y', hint: 'Ctrl+Q' }, false)).not.toBeNull();
  });

  test('rejects a hint whose matcher refuses the chord and which is not native', () => {
    const bindings = {
      fake: {
        mac: { hint: '⌘K', matcher: () => false },
        other: { hint: 'Ctrl+K', native: true },
      },
    };
    expect(hintProblem({ place: 'x', label: 'y', hint: '⌘K' }, true, bindings)).not.toBeNull();
    expect(hintProblem({ place: 'x', label: 'y', hint: 'Ctrl+K' }, false, bindings)).toBeNull();
  });

  test('renders hint cells from submenus too', () => {
    const cells = renderedHints('x', [
      { label: 'Outer', submenu: [{ label: 'Inner', hint: 'Ctrl+\\', run: () => {} }] },
    ]);
    expect(cells).toEqual([{ place: 'x › Outer', label: 'Inner', hint: 'Ctrl+\\' }]);
  });
});
