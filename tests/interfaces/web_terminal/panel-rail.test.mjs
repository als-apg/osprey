/**
 * Unit tests for the sidebar's panel rows (DOM renderer).
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-rail.test.mjs
 *
 * panel-rail.js is a pure DOM module: it places the sidebar's panel rows in
 * the operator's groups (sidebar-groups.js: Workspace and Tools until the
 * operator changes them) and exposes imperative mutators (active / enabled /
 * attention / popped out / non-destructive append, place & remove) that
 * panel-manager's state machine drives. These tests pin that DOM contract —
 * selectors, data-* attributes, class hooks, callback wiring — the same
 * contract the Playwright browser suite selects on.
 *
 * The rows are a MEMBERSHIP list: a row is in view iff the panel is a member.
 * There is no dimmed/closed state — a built row leaves the DOM, a page-owned
 * row (shipped in index.html) is parked `hidden` in place.
 *
 * The rows carry NO per-panel health readout: backend liveness is reported by
 * the SYSTEM panel's `web_panels` health category. `.disabled` and
 * `.unreachable` are the only health-derived states a row renders.
 *
 * Two fixtures: a bare element (no skeleton, so the module builds the groups
 * and plain rows itself) and the `#panel-rail` subtree sliced verbatim out of
 * the shipped index.html (page-owned rows adopted, never rebuilt). The groups
 * model is the real one, kept in this test's localStorage, which is cleared
 * before every test.
 *
 * Imported by RELATIVE path — this module lives under web_terminal; the
 * design-system glyphs come through the /design-system/js/* alias. The
 * environment is happy-dom (vitest.config.js), so `document` is a global.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  SIDEBAR_RENDERED_EVENT,
  createRail,
  addEntry,
  removeEntry,
  entryIds,
  placeEntry,
  getEntry,
  setActive,
  setEntryEnabled,
  setEntryAttention,
  setEntryStatus,
} from '../../../src/osprey/interfaces/web_terminal/static/js/panel-rail.js';
import {
  addGroup,
  groupOf,
  moveToGroup,
  removeGroup,
  renameGroup,
  reorderGroup,
} from '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-groups.js';
import { icon } from '/design-system/js/icons.js';

const STATIC_DIR = join(import.meta.dirname, '../../../src/osprey/interfaces/web_terminal/static');
const indexHtml = readFileSync(join(STATIC_DIR, 'index.html'), 'utf8');

const PANELS = [
  { id: 'artifacts', label: 'Artifacts' },
  { id: 'ariel', label: 'Ariel' },
  { id: 'channel-finder', label: 'Channels' },
];

/** A bare rail element: no skeleton, so createRail builds the groups.
 *  @returns {HTMLElement} */
function freshRail() {
  document.body.innerHTML = '';
  const el = document.createElement('div');
  document.body.appendChild(el);
  return el;
}

/**
 * The `<div … id="panel-rail">…</div>` subtree, verbatim from the shipped
 * template, by counting <div> depth from its opening tag.
 * @returns {string}
 */
function railMarkup() {
  const start = indexHtml.indexOf('<div class="sidebar-sections" id="panel-rail"');
  expect(start, 'index.html has no #panel-rail').toBeGreaterThan(-1);
  const tags = /<div\b[^>]*>|<\/div\s*>/g;
  tags.lastIndex = start;
  let depth = 0;
  for (let m = tags.exec(indexHtml); m; m = tags.exec(indexHtml)) {
    depth += m[0].startsWith('</') ? -1 : 1;
    if (depth === 0) return indexHtml.slice(start, m.index + m[0].length);
  }
  throw new Error('unbalanced #panel-rail in index.html');
}

/** The shipped rail, page-owned rows and all. @returns {HTMLElement} */
function pageRail() {
  document.body.innerHTML = railMarkup();
  return /** @type {HTMLElement} */ (document.getElementById('panel-rail'));
}

/** Every row button in the DOM, parked or not, top to bottom.
 *  @param {HTMLElement} rail @returns {(string | null)[]} */
function allRowIds(rail) {
  return [...rail.querySelectorAll('.panel-rail-button')].map((b) =>
    b.getAttribute('data-panel-id')
  );
}

/** @param {HTMLElement} rail @param {string} name @returns {HTMLElement} */
function group(rail, name) {
  return /** @type {HTMLElement} */ (
    rail.querySelector(`:scope > .sidebar-group[data-group="${name}"]`)
  );
}

/** The direct children of a group, by heading / id / data-panel-id; the
 *  empty line reads `empty` while it shows and is left out while hidden.
 *  @param {HTMLElement} groupEl @returns {string[]} */
function childKeys(groupEl) {
  return [...groupEl.children].flatMap((el) => {
    if (el.classList.contains('osprey-section-label')) {
      return [`label:${el.querySelector('.sidebar-group-name')?.textContent}`];
    }
    if (el.hasAttribute('data-group-empty')) {
      return /** @type {HTMLElement} */ (el).hidden ? [] : ['empty'];
    }
    return [el.id || String(el.getAttribute('data-panel-id'))];
  });
}

/** The group ids of a rail, top to bottom.
 *  @param {HTMLElement} rail @returns {(string | null)[]} */
function groupOrder(rail) {
  return [...rail.querySelectorAll(':scope > .sidebar-group')].map((g) =>
    g.getAttribute('data-group')
  );
}

/** @param {HTMLElement} rail @param {string} id @returns {HTMLButtonElement} */
function row(rail, id) {
  return /** @type {HTMLButtonElement} */ (getEntry(rail, id));
}

/** Count `osprey-sidebar-rendered` dispatches from now on.
 *  @returns {{ count: number, stop: () => void }} */
function countRenders() {
  const seen = { count: 0, stop: () => {} };
  const on = () => {
    seen.count += 1;
  };
  document.addEventListener(SIDEBAR_RENDERED_EVENT, on);
  seen.stop = () => document.removeEventListener(SIDEBAR_RENDERED_EVENT, on);
  return seen;
}

// The groups live in this browser's storage; every test starts from the
// default Workspace and Tools.
beforeEach(() => {
  localStorage.clear();
});

describe('createRail', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
  });

  test('renders one row per panel, in order, with data-panel-id', () => {
    createRail(rail, PANELS);
    expect(entryIds(rail)).toEqual(['artifacts', 'ariel', 'channel-finder']);
  });

  test('marks the container as a list of labelled groups, the built-in two by default', () => {
    createRail(rail, PANELS);
    expect(rail.getAttribute('role')).toBe('list');
    const groups = [...rail.children];
    expect(groups.map((g) => g.getAttribute('data-group'))).toEqual(['workspace', 'tools']);
    for (const g of groups) {
      expect(g.classList.contains('sidebar-group')).toBe(true);
      expect(g.getAttribute('role')).toBe('listitem');
      const heading = /** @type {HTMLElement} */ (g.firstElementChild);
      expect(heading.classList.contains('osprey-section-label')).toBe(true);
      const name = /** @type {HTMLElement} */ (heading.querySelector('.sidebar-group-name'));
      expect(g.getAttribute('aria-labelledby')).toBe(name.id);
      expect(heading.querySelector('button.osprey-mini[data-group-menu]')).not.toBeNull();
    }
    expect(groups.map((g) => g.querySelector('.sidebar-group-name')?.textContent)).toEqual([
      'Workspace',
      'Tools',
    ]);
  });

  test('Artifacts sits under Workspace and every other panel under Tools until moved', () => {
    createRail(rail, PANELS);
    expect(childKeys(group(rail, 'workspace'))).toEqual(['label:Workspace', 'artifacts']);
    expect(childKeys(group(rail, 'tools'))).toEqual(['label:Tools', 'ariel', 'channel-finder']);
  });

  test('no row is a tab: no tab roles, no selection state, no close corner', () => {
    createRail(rail, PANELS);
    expect(rail.querySelector('[role="tab"]')).toBeNull();
    expect(rail.querySelector('[aria-selected]')).toBeNull();
    expect(rail.querySelector('.panel-rail-close')).toBeNull();
  });

  test('each row carries a glyph and a label', () => {
    createRail(rail, PANELS);
    const first = row(rail, 'artifacts');
    expect(first.classList.contains('osprey-nav-row')).toBe(true);
    expect(first.querySelector('svg.osprey-ic.osprey-nav-icon')).not.toBeNull();
    expect(first.querySelector('.osprey-nav-label')?.textContent).toBe('Artifacts');
  });

  test('the glyph is the panel icon when named, else the workspace glyph', () => {
    createRail(rail, [
      { id: 'ariel', label: 'Ariel', icon: 'book-open' },
      { id: 'x', label: 'X' },
    ]);
    expect(row(rail, 'ariel').querySelector('.osprey-nav-icon')?.innerHTML).toBe(
      icon('book-open').innerHTML
    );
    expect(row(rail, 'x').querySelector('.osprey-nav-icon')?.innerHTML).toBe(
      icon('workspace').innerHTML
    );
  });

  test('rows start disabled and not current', () => {
    createRail(rail, PANELS);
    const first = row(rail, 'artifacts');
    expect(first.classList.contains('disabled')).toBe(true);
    expect(first.classList.contains('active')).toBe(false);
    expect(first.hasAttribute('aria-current')).toBe(false);
    expect(first.getAttribute('title')).toBe('Artifacts');
    expect(first.getAttribute('aria-label')).toBe('Artifacts');
    expect(first.dataset.tip).toBe('Artifacts');
  });

  test('a hint is appended to the tooltip, never to the accessible name or the label tip', () => {
    // The context menu has no visible affordance, so the tooltip is where it
    // is advertised; aria-label stays the panel's identity so a screen reader
    // does not read the advertisement on every row.
    createRail(rail, [{ id: 'ariel', label: 'Ariel', hint: 'right-click for actions' }]);
    const entry = row(rail, 'ariel');
    expect(entry.getAttribute('title')).toBe('Ariel · right-click for actions');
    expect(entry.getAttribute('aria-label')).toBe('Ariel');
    expect(entry.dataset.tip).toBe('Ariel');
  });

  test("a row's tooltip detail is the panel's sentence; a custom panel has none", () => {
    createRail(rail, [
      { id: 'ariel', label: 'Ariel' },
      { id: 'my-dashboard', label: 'My dashboard' },
    ]);
    expect(row(rail, 'ariel').dataset.tipDetail).toBe('Search the electronic logbook');
    expect(row(rail, 'my-dashboard').hasAttribute('data-tip-detail')).toBe(false);
  });

  test('label is set via textContent (no HTML injection)', () => {
    createRail(rail, [{ id: 'x', label: '<img src=x onerror=alert(1)>' }]);
    const entry = row(rail, 'x');
    expect(entry.querySelector('.osprey-nav-label')?.textContent).toBe(
      '<img src=x onerror=alert(1)>'
    );
    expect(entry.querySelector('img')).toBeNull();
  });

  test('full render replaces prior rows and keeps the two groups', () => {
    createRail(rail, PANELS);
    createRail(rail, [{ id: 'okf', label: 'Knowledge' }]);
    expect(entryIds(rail)).toEqual(['okf']);
    expect(rail.querySelectorAll('.sidebar-group').length).toBe(2);
  });

  test('every render announces itself on document', () => {
    const renders = countRenders();
    createRail(rail, PANELS);
    createRail(rail, PANELS);
    renders.stop();
    expect(SIDEBAR_RENDERED_EVENT).toBe('osprey-sidebar-rendered');
    expect(renders.count).toBe(2);
  });
});

/**
 * The shipped page carries the Agent row, the Artifacts row (wrapping its
 * lists), the Script runs row and `+ Add panel`. A render MOVES them into
 * place; it never rebuilds them, so what other modules hang on them survives.
 */
describe('page-owned rows', () => {
  /** @type {HTMLElement} */
  let rail;
  /** @type {HTMLElement} */
  let agentRow;
  /** @type {HTMLElement} */
  let artifactsRow;
  /** @type {HTMLElement} */
  let artifactsButton;
  /** @type {HTMLElement} */
  let runsRow;
  /** @type {HTMLElement} */
  let addButton;

  const ALL = [
    { id: 'ariel', label: 'Ariel' },
    { id: 'events', label: 'Events' },
    { id: 'terminal', label: 'Agent' },
    { id: 'lattice', label: 'Lattice' },
    { id: 'artifacts', label: 'Artifacts' },
  ];

  beforeEach(() => {
    rail = pageRail();
    agentRow = /** @type {HTMLElement} */ (document.getElementById('sidebar-agent-row'));
    artifactsRow = /** @type {HTMLElement} */ (document.getElementById('sidebar-artifacts-row'));
    artifactsButton = /** @type {HTMLElement} */ (
      artifactsRow.querySelector('.panel-rail-button[data-panel-id="artifacts"]')
    );
    runsRow = /** @type {HTMLElement} */ (document.getElementById('sidebar-runs-row'));
    addButton = /** @type {HTMLElement} */ (document.getElementById('panel-add-btn'));
  });

  test('the shipped skeleton holds every node the renderer adopts', () => {
    for (const el of [agentRow, artifactsRow, artifactsButton, runsRow, addButton]) {
      expect(el).not.toBeNull();
    }
    expect(agentRow.getAttribute('data-panel-id')).toBe('terminal');
    expect(groupOrder(rail)).toEqual(['workspace', 'tools']);
  });

  test('the shipped groups and headings are adopted, never rebuilt', () => {
    const workspace = group(rail, 'workspace');
    const heading = /** @type {HTMLElement} */ (workspace.firstElementChild);
    createRail(rail, ALL);
    renameGroup('workspace', 'Operations');

    expect(group(rail, 'workspace')).toBe(workspace);
    expect(workspace.firstElementChild).toBe(heading);
    const name = /** @type {HTMLElement} */ (heading.querySelector('.sidebar-group-name'));
    expect(name.id).toBe('sidebar-workspace-label');
    expect(name.textContent).toBe('Operations');
    expect(workspace.getAttribute('aria-labelledby')).toBe('sidebar-workspace-label');
  });

  test('the Agent row leads the first group, and each group keeps the rail order', () => {
    createRail(rail, ALL);
    expect(childKeys(group(rail, 'workspace'))).toEqual([
      'label:Workspace',
      'sidebar-agent-row',
      'events',
      'sidebar-artifacts-row',
      'sidebar-runs-row',
    ]);
    expect(childKeys(group(rail, 'tools'))).toEqual([
      'label:Tools',
      'ariel',
      'lattice',
      'panel-add-btn',
    ]);
    expect(entryIds(rail)).toEqual(['terminal', 'events', 'artifacts', 'ariel', 'lattice']);
  });

  test('the Artifacts row moves to another group as the same node, Script runs with it', () => {
    const kids = document.getElementById('sidebar-artifacts-kids');
    createRail(rail, ALL);
    setActive(rail, 'artifacts');

    moveToGroup('artifacts', 'tools');

    expect(childKeys(group(rail, 'workspace'))).toEqual([
      'label:Workspace',
      'sidebar-agent-row',
      'events',
    ]);
    expect(childKeys(group(rail, 'tools'))).toEqual([
      'label:Tools',
      'ariel',
      'lattice',
      'sidebar-artifacts-row',
      'sidebar-runs-row',
      'panel-add-btn',
    ]);
    expect(getEntry(rail, 'artifacts')).toBe(artifactsButton);
    expect(document.getElementById('sidebar-artifacts-kids')).toBe(kids);
    expect(artifactsButton.classList.contains('active')).toBe(true);
  });

  test('+ Add panel closes the last group, whichever group that is', () => {
    createRail(rail, ALL);
    const id = addGroup('Beamline');
    expect(childKeys(group(rail, id))).toEqual(['label:Beamline', 'empty', 'panel-add-btn']);

    reorderGroup('workspace', 'down');
    reorderGroup('workspace', 'down');
    expect(groupOrder(rail)).toEqual(['tools', id, 'workspace']);
    expect(group(rail, 'workspace').lastElementChild).toBe(addButton);
  });

  test('a group whose rows are all parked shows its empty line', () => {
    createRail(rail, [{ id: 'ariel', label: 'Ariel' }]);
    // The parked rows stay in their group, hidden, so it reads as empty.
    expect(childKeys(group(rail, 'workspace'))).toEqual([
      'label:Workspace',
      'empty',
      'sidebar-agent-row',
      'sidebar-artifacts-row',
      'sidebar-runs-row',
    ]);
    expect([agentRow, artifactsRow, runsRow].every((el) => el.hidden)).toBe(true);
    expect(childKeys(group(rail, 'tools'))).toEqual(['label:Tools', 'ariel', 'panel-add-btn']);
  });

  test('the page-owned nodes are adopted, never rebuilt, across renders', () => {
    const kids = document.getElementById('sidebar-artifacts-kids');
    createRail(rail, ALL);
    createRail(rail, ALL);
    expect(getEntry(rail, 'terminal')).toBe(agentRow);
    expect(getEntry(rail, 'artifacts')).toBe(artifactsButton);
    expect(document.getElementById('sidebar-artifacts-row')).toBe(artifactsRow);
    expect(document.getElementById('sidebar-artifacts-kids')).toBe(kids);
    expect(document.getElementById('sidebar-runs-row')).toBe(runsRow);
    expect(document.getElementById('panel-add-btn')).toBe(addButton);
    expect(rail.querySelectorAll('#sidebar-agent-row').length).toBe(1);
    expect(rail.querySelectorAll('[data-panel-id="artifacts"]').length).toBe(1);
  });

  test('an adopted row takes the identity it is rendered with', () => {
    createRail(rail, [{ id: 'terminal', label: 'Agent', hint: 'right-click for actions' }]);
    expect(agentRow.title).toBe('Agent · right-click for actions');
    expect(agentRow.getAttribute('aria-label')).toBe('Agent');
    expect(agentRow.dataset.tip).toBe('Agent');
    expect(agentRow.querySelector('.osprey-nav-label')?.textContent).toBe('Agent');
  });

  test('an adopted row answers to the latest render, wired once', () => {
    const first = vi.fn();
    const second = vi.fn();
    createRail(rail, ALL, { onActivate: first });
    createRail(rail, ALL, { onActivate: second });

    agentRow.click();
    artifactsButton.click();

    expect(first).not.toHaveBeenCalled();
    expect(second.mock.calls).toEqual([['terminal'], ['artifacts']]);
  });

  test('a page-owned row whose panel is absent is parked hidden and counts as absent', () => {
    createRail(rail, [{ id: 'ariel', label: 'Ariel' }]);
    expect(agentRow.isConnected).toBe(true);
    expect(agentRow.hidden).toBe(true);
    expect(artifactsRow.hidden).toBe(true);
    expect(getEntry(rail, 'terminal')).toBeNull();
    expect(getEntry(rail, 'artifacts')).toBeNull();
    expect(entryIds(rail)).toEqual(['ariel']);
  });

  test('the Script runs row is shown exactly while Artifacts is', () => {
    createRail(rail, [{ id: 'terminal', label: 'Agent' }]);
    expect(runsRow.hidden).toBe(true);

    createRail(rail, ALL);
    expect(runsRow.hidden).toBe(false);
  });

  test('removeEntry parks a page-owned row in place and hides the Script runs row', () => {
    createRail(rail, ALL);
    const renders = countRenders();

    removeEntry(rail, 'artifacts');
    renders.stop();

    expect(artifactsRow.isConnected).toBe(true);
    expect(artifactsRow.hidden).toBe(true);
    expect(runsRow.hidden).toBe(true);
    expect(getEntry(rail, 'artifacts')).toBeNull();
    expect(renders.count).toBe(1);
  });

  test('addEntry brings a parked page-owned row back, Script runs row with it', () => {
    createRail(rail, [{ id: 'terminal', label: 'Agent' }]);
    const renders = countRenders();

    const entry = addEntry(rail, { id: 'artifacts', label: 'Artifacts' });
    renders.stop();

    expect(entry).toBe(artifactsButton);
    expect(artifactsRow.hidden).toBe(false);
    expect(runsRow.hidden).toBe(false);
    expect(renders.count).toBe(1);
    expect(entryIds(rail)).toEqual(['terminal', 'artifacts']);
  });

  test('removing and re-adding the Agent row keeps the same node', () => {
    createRail(rail, ALL);
    removeEntry(rail, 'terminal');
    expect(agentRow.hidden).toBe(true);
    expect(getEntry(rail, 'terminal')).toBeNull();

    expect(addEntry(rail, { id: 'terminal', label: 'Agent' })).toBe(agentRow);
    expect(agentRow.hidden).toBe(false);
  });

  test('a built Tools row added at runtime lands just above + Add panel', () => {
    createRail(rail, ALL);
    addEntry(rail, { id: 'okf', label: 'Knowledge' });
    expect(childKeys(group(rail, 'tools'))).toEqual([
      'label:Tools',
      'ariel',
      'lattice',
      'okf',
      'panel-add-btn',
    ]);
  });

  test('a built row added at runtime joins the foot of its group', () => {
    createRail(rail, [
      { id: 'terminal', label: 'Agent' },
      { id: 'artifacts', label: 'Artifacts' },
    ]);
    addEntry(rail, { id: 'events', label: 'Events' });
    expect(entryIds(rail)).toEqual(['terminal', 'artifacts', 'events']);
    expect(group(rail, 'workspace').contains(getEntry(rail, 'events'))).toBe(true);
  });

  test('the attention dot sits before the disclosure chevron of a parent row', () => {
    createRail(rail, ALL);
    setEntryAttention(rail, 'artifacts', true);
    const dot = /** @type {HTMLElement} */ (
      artifactsButton.querySelector(':scope > [data-attention-badge]')
    );
    expect(dot.nextElementSibling?.classList.contains('osprey-nav-disc')).toBe(true);
  });

  test('setActive marks a page-owned row current', () => {
    createRail(rail, ALL);
    setActive(rail, 'terminal');
    expect(agentRow.classList.contains('active')).toBe(true);
    expect(agentRow.getAttribute('aria-current')).toBe('page');
  });

  test('a render under a borrowed tooltip retitles the stash, so the clear restores it', () => {
    // An adopted row keeps its live state across renders, a status message
    // included; the identity the render brings must be what the clear puts back.
    createRail(rail, ALL);
    setEntryStatus(rail, 'terminal', { failed: false, message: 'starting' });

    createRail(rail, [{ id: 'terminal', label: 'Agent', hint: 'right-click for actions' }]);
    expect(agentRow.title).toBe('starting');

    setEntryStatus(rail, 'terminal', { failed: false, message: null });
    expect(agentRow.title).toBe('Agent · right-click for actions');
    expect(agentRow.hasAttribute('data-title-base')).toBe(false);
  });
});

/**
 * The groups come from the operator's model (sidebar-groups.js). A change to
 * the model moves the rows at once — the same nodes, with their live state —
 * and announces a render, so the modules that fill the page-owned rows follow.
 */
describe('groups', () => {
  const ROWS = [
    { id: 'terminal', label: 'Agent' },
    { id: 'artifacts', label: 'Artifacts' },
    { id: 'ariel', label: 'Ariel' },
    { id: 'events', label: 'Events' },
    { id: 'lattice', label: 'Lattice' },
    { id: 'okf', label: 'Knowledge' },
  ];
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
  });

  test('the groups render in the model order, a group of the operator’s own among them', () => {
    const id = addGroup('Beamline', 'workspace');
    renameGroup('tools', 'Instruments');
    createRail(rail, ROWS);

    expect(groupOrder(rail)).toEqual(['workspace', id, 'tools']);
    expect(childKeys(group(rail, id))).toEqual(['label:Beamline', 'empty']);
    expect(childKeys(group(rail, 'tools'))).toEqual([
      'label:Instruments',
      'ariel',
      'lattice',
      'okf',
    ]);
  });

  test('a moved row joins its new group in rail order, as the same node with its state', () => {
    createRail(rail, ROWS);
    const okf = row(rail, 'okf');
    setActive(rail, 'okf');
    setEntryEnabled(rail, 'okf', true);
    const renders = countRenders();

    moveToGroup('okf', 'workspace');
    moveToGroup('ariel', 'workspace');
    renders.stop();

    expect(childKeys(group(rail, 'workspace'))).toEqual([
      'label:Workspace',
      'terminal',
      'artifacts',
      'ariel',
      'events',
      'okf',
    ]);
    expect(row(rail, 'okf')).toBe(okf);
    expect(okf.classList.contains('active')).toBe(true);
    expect(okf.classList.contains('disabled')).toBe(false);
    expect(renders.count).toBe(2);
  });

  test('a group that loses its last row shows its empty line, and hides it again', () => {
    const id = addGroup('Beamline');
    createRail(rail, ROWS);
    moveToGroup('lattice', id);
    expect(childKeys(group(rail, id))).toEqual(['label:Beamline', 'lattice']);

    moveToGroup('lattice', 'tools');
    expect(childKeys(group(rail, id))).toEqual(['label:Beamline', 'empty']);
    const line = /** @type {HTMLElement} */ (group(rail, id).querySelector('[data-group-empty]'));
    expect(line.classList.contains('osprey-list-empty')).toBe(true);
    expect(line.textContent).toBe("No panels — move one here from a row's menu");
  });

  test('the Agent row stays first in whichever group is first', () => {
    createRail(rail, ROWS);
    reorderGroup('tools', 'up');

    expect(groupOrder(rail)).toEqual(['tools', 'workspace']);
    expect(childKeys(group(rail, 'tools'))).toEqual([
      'label:Tools',
      'terminal',
      'ariel',
      'lattice',
      'okf',
    ]);
    expect(childKeys(group(rail, 'workspace'))).toEqual(['label:Workspace', 'artifacts', 'events']);
    expect(entryIds(rail)[0]).toBe('terminal');
  });

  test('a deleted group gives its rows back to Tools and leaves the rail', () => {
    const id = addGroup('Beamline');
    createRail(rail, ROWS);
    moveToGroup('artifacts', id);
    moveToGroup('lattice', id);

    removeGroup(id);

    expect(groupOrder(rail)).toEqual(['workspace', 'tools']);
    expect(childKeys(group(rail, 'tools'))).toEqual([
      'label:Tools',
      'artifacts',
      'ariel',
      'lattice',
      'okf',
    ]);
  });

  test('a rail no longer in the document stops following the groups', () => {
    createRail(rail, ROWS);
    rail.remove();
    moveToGroup('ariel', 'workspace');
    expect(group(rail, 'tools').contains(getEntry(rail, 'ariel'))).toBe(true);
  });
});

describe('row interactions', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
  });

  test('clicking a row invokes onActivate with its id', () => {
    /** @type {string[]} */
    const activated = [];
    createRail(rail, PANELS, { onActivate: (id) => activated.push(id) });
    row(rail, 'ariel').click();
    expect(activated).toEqual(['ariel']);
  });

  test('right-click forwards (id, event) to onContextMenu and suppresses the browser menu', () => {
    /** @type {any[]} */
    const calls = [];
    createRail(rail, PANELS, {
      onContextMenu: (id, e) => {
        calls.push([id, e]);
        return true;
      },
    });
    const ev = new MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      clientX: 12,
      clientY: 34,
    });
    row(rail, 'ariel').dispatchEvent(ev);

    expect(calls).toEqual([['ariel', ev]]);
    expect(ev.defaultPrevented).toBe(true);
  });

  test('a declined right-click keeps the native browser menu', () => {
    // The caller's policy — a disabled row, say — is expressed as `false`,
    // and the sidebar must then leave the event alone.
    createRail(rail, PANELS, { onContextMenu: () => false });
    const ev = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    row(rail, 'ariel').dispatchEvent(ev);

    expect(ev.defaultPrevented).toBe(false);
  });
});

/**
 * The keyboard route to the context menu. The sidebar opens the menu itself
 * rather than waiting for the platform: macOS fires no `contextmenu` event
 * from the keyboard at all. Where the platform DOES fire one (Windows menu
 * key, Firefox Shift+F10) it is the keydown's default action, so the same
 * boolean contract as right-click decides whether that default is suppressed
 * — the difference between one menu and two.
 */
describe('keyboard route to the context menu', () => {
  /** @type {HTMLElement} */
  let rail;

  // happy-dom reports an all-zero rect, so the row's geometry is stubbed:
  // the assertion is about WHERE the menu is asked to open, not about layout.
  const RECT = /** @type {DOMRect} */ (
    /** @type {unknown} */ ({
      x: 10,
      y: 40,
      left: 10,
      top: 40,
      width: 200,
      height: 30,
      right: 210,
      bottom: 70,
      toJSON: () => ({}),
    })
  );
  const CENTER_X = 110; // 10 + 200 / 2
  const CENTER_Y = 55; // 40 + 30 / 2

  /** @param {string} key @param {boolean} [shiftKey] */
  function keyEvent(key, shiftKey = false) {
    return new KeyboardEvent('keydown', { key, shiftKey, bubbles: true, cancelable: true });
  }

  /** Wire a rail whose onContextMenu records its calls and reports `handled`.
   *  @param {boolean} handled @returns {any[]} */
  function railWithMenu(handled) {
    /** @type {any[]} */
    const calls = [];
    createRail(rail, PANELS, {
      onContextMenu: (id, e) => {
        calls.push([id, e]);
        return handled;
      },
    });
    return calls;
  }

  beforeEach(() => {
    rail = freshRail();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test.each([
    ['the ContextMenu key', 'ContextMenu', false],
    ['Shift+F10', 'F10', true],
  ])('%s opens the menu centred on the row rect', (_name, key, shift) => {
    const calls = railWithMenu(true);
    const entry = row(rail, 'ariel');
    vi.spyOn(entry, 'getBoundingClientRect').mockReturnValue(RECT);

    const ev = keyEvent(key, shift);
    entry.dispatchEvent(ev);

    // Exactly one open: the synthetic event is handed to the caller, never
    // dispatched, so the row's own contextmenu listener cannot also fire.
    expect(calls.length).toBe(1);
    const [id, menuEvent] = calls[0];
    expect(id).toBe('ariel');
    expect(menuEvent.type).toBe('contextmenu');
    expect(menuEvent.clientX).toBe(CENTER_X);
    expect(menuEvent.clientY).toBe(CENTER_Y);
    // Windows/Firefox would otherwise open a second, native menu on this key.
    expect(ev.defaultPrevented).toBe(true);
  });

  test('a declined row keeps the platform default on the keyboard too', () => {
    const calls = railWithMenu(false);
    row(rail, 'ariel').dispatchEvent(keyEvent('ContextMenu'));

    expect(calls.length).toBe(1);
    // No menu was opened, so whatever the platform does with this key stands.
    expect(calls[0][1].defaultPrevented).toBe(false);
  });

  test('F10 without Shift, and ordinary keys, are left alone', () => {
    const calls = railWithMenu(true);
    const entry = row(rail, 'ariel');

    for (const ev of [keyEvent('F10'), keyEvent('Enter'), keyEvent('m', true)]) {
      entry.dispatchEvent(ev);
      expect(ev.defaultPrevented).toBe(false);
    }
    expect(calls).toEqual([]);
  });
});

describe('drag from the sidebar (onDragStart / onDragEnd)', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
  });

  /** Build a dragstart-shaped event happy-dom can dispatch (no DragEvent there).
   *  @param {string} type */
  function dragEvent(type) {
    const ev = new Event(type, { bubbles: true, cancelable: true });
    /** @type {any} */ (ev).dataTransfer = {
      data: /** @type {Record<string, string>} */ ({}),
      effectAllowed: '',
      setData(/** @type {string} */ k, /** @type {string} */ v) {
        this.data[k] = v;
      },
    };
    return ev;
  }

  test('rows are draggable', () => {
    createRail(rail, PANELS, { onDragStart: () => true });
    expect(row(rail, 'ariel').getAttribute('draggable')).toBe('true');
  });

  test('dragstart hands (id, dataTransfer) to onDragStart and marks the source row', () => {
    /** @type {any[]} */
    const calls = [];
    createRail(rail, PANELS, {
      onDragStart: (id, dt) => {
        calls.push([id, dt]);
        return true;
      },
    });
    const entry = row(rail, 'ariel');
    const ev = dragEvent('dragstart');
    entry.dispatchEvent(ev);

    expect(calls).toEqual([['ariel', /** @type {any} */ (ev).dataTransfer]]);
    expect(entry.classList.contains('dragging')).toBe(true);
    expect(ev.defaultPrevented).toBe(false);
  });

  test('onDragStart returning false cancels the drag (no dragging state, default prevented)', () => {
    createRail(rail, PANELS, { onDragStart: () => false });
    const entry = row(rail, 'ariel');
    const ev = dragEvent('dragstart');
    entry.dispatchEvent(ev);

    expect(entry.classList.contains('dragging')).toBe(false);
    expect(ev.defaultPrevented).toBe(true);
  });

  test('removing a mid-drag row ends its drag gesture (removeEntry heals the shields)', () => {
    // A detached drag source can never fire dragend (HTML5 delivers it to the
    // source element only), so the caller's onDragEnd — which lowers the
    // iframe pointer shields — would never run and every panel would stay
    // frozen. removeEntry must end the gesture itself before detaching.
    /** @type {string[]} */
    const ended = [];
    createRail(rail, PANELS, {
      onDragStart: () => true,
      onDragEnd: (id) => ended.push(id),
    });
    const entry = row(rail, 'ariel');
    entry.dispatchEvent(dragEvent('dragstart'));
    expect(entry.classList.contains('dragging')).toBe(true);

    removeEntry(rail, 'ariel');

    expect(getEntry(rail, 'ariel')).toBeNull();
    expect(ended).toEqual(['ariel']);
  });

  test('removeEntry of an idle row does not fire onDragEnd', () => {
    /** @type {string[]} */
    const ended = [];
    createRail(rail, PANELS, {
      onDragStart: () => true,
      onDragEnd: (id) => ended.push(id),
    });

    removeEntry(rail, 'ariel');

    expect(ended).toEqual([]);
  });

  test('dragend clears the dragging state and calls onDragEnd', () => {
    /** @type {string[]} */
    const ended = [];
    createRail(rail, PANELS, {
      onDragStart: () => true,
      onDragEnd: (id) => ended.push(id),
    });
    const entry = row(rail, 'ariel');
    entry.dispatchEvent(dragEvent('dragstart'));
    expect(entry.classList.contains('dragging')).toBe(true);

    entry.dispatchEvent(dragEvent('dragend'));
    expect(entry.classList.contains('dragging')).toBe(false);
    expect(ended).toEqual(['ariel']);
  });
});

/**
 * A row dragged within the sidebar reorders its group, or moves to another
 * group and takes its place there. The Agent row is pinned: it is neither
 * dragged nor a target, and a row the caller marks fixed (`reorderable`
 * false) is neither either.
 */
describe('reorder and move by drag (onReorder)', () => {
  const ROW = [
    { id: 'terminal', label: 'Agent' },
    { id: 'artifacts', label: 'Artifacts' },
    { id: 'ariel', label: 'Ariel' },
    { id: 'channel-finder', label: 'Channels' },
    { id: 'lattice', label: 'Lattice' },
    { id: 'okf', label: 'Knowledge' },
  ];
  /** @type {HTMLElement} */
  let rail;
  /** @type {[string, string | null][]} */
  let moves;

  /**
   * happy-dom has no layout: stack every heading (24 px) and every row in
   * view (40 px) top to bottom, in DOM order. Every row shares one column,
   * so x never decides anything. With the default groups that puts the
   * Workspace heading at 0, the Agent row at 24, Artifacts at 64, the Tools
   * heading at 104, then ariel 128, channel-finder 168, lattice 208 and
   * okf 248.
   */
  function layOut() {
    let top = 0;
    for (const g of rail.querySelectorAll(':scope > .sidebar-group')) {
      for (const el of g.children) {
        const node = /** @type {HTMLElement} */ (el);
        if (node.hidden) continue;
        const height = node.classList.contains('osprey-section-label') ? 24 : 40;
        const rect = /** @type {DOMRect} */ ({ top, height, left: 0, width: 200 });
        node.getBoundingClientRect = () => rect;
        top += height;
      }
    }
  }

  beforeEach(() => {
    rail = freshRail();
    moves = [];
    createRail(rail, ROW, {
      onDragStart: () => true,
      onReorder: (id, before) => moves.push([id, before]),
      reorderable: (id) => id !== 'okf',
    });
    layOut();
  });

  /** @param {string} type @param {number} y  pointer height over the sidebar */
  function pointerEvent(type, y) {
    const ev = new Event(type, { bubbles: true, cancelable: true });
    Object.assign(ev, { clientX: 100, clientY: y, dataTransfer: { dropEffect: '' } });
    return ev;
  }

  /** Start dragging `id`'s row. @param {string} id */
  function grab(id) {
    row(rail, id).dispatchEvent(
      Object.assign(new Event('dragstart', { bubbles: true, cancelable: true }), {
        dataTransfer: { setData() {} },
      })
    );
  }

  /** The ids carrying the insertion line. */
  function marked() {
    return [...rail.querySelectorAll('.drop-before, .drop-after')].map((el) => [
      el.getAttribute('data-panel-id'),
      el.classList.contains('drop-before') ? 'before' : 'after',
    ]);
  }

  test('dropping above a row of its own group moves the dragged one above it', () => {
    grab('lattice');
    const over = pointerEvent('dragover', 140); // upper half of ariel (128–168)
    rail.dispatchEvent(over);
    expect(over.defaultPrevented).toBe(true);
    expect(marked()).toEqual([['ariel', 'before']]);

    rail.dispatchEvent(pointerEvent('drop', 140));
    expect(moves).toEqual([['lattice', 'ariel']]);
    expect(groupOf('lattice')).toBe('tools');
    expect(marked()).toEqual([]);
  });

  test('dropping past the last row puts the dragged one at the foot of its group', () => {
    grab('ariel');
    rail.dispatchEvent(pointerEvent('dragover', 280)); // over the fixed okf row
    expect(marked()).toEqual([['okf', 'after']]);
    rail.dispatchEvent(pointerEvent('drop', 280));
    expect(moves).toEqual([['ariel', null]]);
  });

  test("a drop on a group's heading lands at the head of that group", () => {
    grab('channel-finder');
    rail.dispatchEvent(pointerEvent('dragover', 110));
    expect(marked()).toEqual([['ariel', 'before']]);
    rail.dispatchEvent(pointerEvent('drop', 110));
    expect(moves).toEqual([['channel-finder', 'ariel']]);
  });

  test('a drop into another group moves the row there first, then places it', () => {
    grab('lattice');
    rail.dispatchEvent(pointerEvent('dragover', 70)); // upper half of artifacts (64–104)
    expect(marked()).toEqual([['artifacts', 'before']]);

    rail.dispatchEvent(pointerEvent('drop', 70));
    expect(groupOf('lattice')).toBe('workspace');
    expect(group(rail, 'workspace').contains(row(rail, 'lattice'))).toBe(true);
    expect(moves).toEqual([['lattice', 'artifacts']]);
  });

  test('a drop at the foot of another group moves the row there, last', () => {
    grab('ariel');
    rail.dispatchEvent(pointerEvent('dragover', 100)); // below the middle of artifacts
    expect(marked()).toEqual([['artifacts', 'after']]);

    rail.dispatchEvent(pointerEvent('drop', 100));
    expect(groupOf('ariel')).toBe('workspace');
    expect(moves).toEqual([['ariel', null]]);
  });

  test('the Agent row is never a target: a drop above it lands after it', () => {
    grab('lattice');
    rail.dispatchEvent(pointerEvent('dragover', 30));
    expect(marked()).toEqual([['artifacts', 'before']]);
  });

  test('an empty group takes a drop, under its empty line', () => {
    const id = addGroup('Beamline');
    layOut(); // the Beamline heading at 288, its empty line at 312
    const line = /** @type {HTMLElement} */ (group(rail, id).querySelector('[data-group-empty]'));
    grab('ariel');

    rail.dispatchEvent(pointerEvent('dragover', 320));
    expect(line.classList.contains('drop-after')).toBe(true);

    rail.dispatchEvent(pointerEvent('drop', 320));
    expect(groupOf('ariel')).toBe(id);
    expect(moves).toEqual([['ariel', null]]);
    expect(line.hidden).toBe(true);
  });

  test('a drop that leaves the row where it is reports no move', () => {
    grab('channel-finder');
    rail.dispatchEvent(pointerEvent('dragover', 175)); // just below its own top
    expect(marked()).toEqual([]);
    rail.dispatchEvent(pointerEvent('drop', 175));
    expect(moves).toEqual([]);
  });

  test.each(['terminal', 'okf'])('the fixed row %s cannot be dragged into a reorder', (id) => {
    grab(id);
    const over = pointerEvent('dragover', 140);
    rail.dispatchEvent(over);
    expect(over.defaultPrevented).toBe(false);
    rail.dispatchEvent(pointerEvent('drop', 140));
    expect(moves).toEqual([]);
  });

  test('a drag that did not start on a row is left alone', () => {
    const over = pointerEvent('dragover', 140);
    rail.dispatchEvent(over);
    expect(over.defaultPrevented).toBe(false);
    rail.dispatchEvent(pointerEvent('drop', 140));
    expect(moves).toEqual([]);
  });

  test('the drop reads the pointer height only', () => {
    grab('lattice');
    const ev = pointerEvent('drop', 140);
    Object.assign(ev, { clientX: 9999 });
    rail.dispatchEvent(ev);
    expect(moves).toEqual([['lattice', 'ariel']]);
  });

  test('dragend clears the insertion line', () => {
    grab('lattice');
    rail.dispatchEvent(pointerEvent('dragover', 140));
    row(rail, 'lattice').dispatchEvent(new Event('dragend', { bubbles: true }));
    expect(marked()).toEqual([]);
  });

  test('a later render re-points the drop at its own onReorder', () => {
    /** @type {[string, string | null][]} */
    const later = [];
    createRail(rail, ROW, {
      onDragStart: () => true,
      onReorder: (id, before) => later.push([id, before]),
    });
    layOut();
    grab('okf');
    rail.dispatchEvent(pointerEvent('drop', 140));
    expect(moves).toEqual([]);
    expect(later).toEqual([['okf', 'ariel']]);
  });
});

describe('addEntry (non-destructive)', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
  });

  test('appends a new Tools row, preserving existing ones', () => {
    createRail(rail, PANELS);
    addEntry(rail, { id: 'lattice', label: 'Lattice' });
    expect(entryIds(rail)).toEqual(['artifacts', 'ariel', 'channel-finder', 'lattice']);
  });

  test('preserves live state on existing rows (no rebuild)', () => {
    createRail(rail, PANELS);
    setActive(rail, 'artifacts');
    setEntryEnabled(rail, 'artifacts', true);

    addEntry(rail, { id: 'lattice', label: 'Lattice' });

    const artifacts = getEntry(rail, 'artifacts');
    expect(artifacts?.classList.contains('active')).toBe(true);
    expect(artifacts?.classList.contains('disabled')).toBe(false);
  });

  test('is idempotent by id — no duplicate node', () => {
    createRail(rail, PANELS);
    const first = addEntry(rail, { id: 'ariel', label: 'Ariel' });
    expect(first).toBe(getEntry(rail, 'ariel'));
    expect(allRowIds(rail).filter((id) => id === 'ariel')).toEqual(['ariel']);
  });

  test('wires onActivate on the appended row', () => {
    /** @type {string[]} */
    const activated = [];
    createRail(rail, PANELS);
    addEntry(rail, { id: 'lattice', label: 'Lattice' }, { onActivate: (id) => activated.push(id) });
    row(rail, 'lattice').click();
    expect(activated).toEqual(['lattice']);
  });

  test('the Agent row added at runtime takes the head of the first group, not the foot', () => {
    createRail(rail, [
      { id: 'events', label: 'Events' },
      { id: 'ariel', label: 'Ariel' },
    ]);
    addEntry(rail, { id: 'terminal', label: 'Agent' });
    expect(entryIds(rail)).toEqual(['terminal', 'events', 'ariel']);
  });
});

describe('removeEntry', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('removes a built row node, preserving the others', () => {
    removeEntry(rail, 'ariel');
    expect(entryIds(rail)).toEqual(['artifacts', 'channel-finder']);
    expect(allRowIds(rail)).toEqual(['artifacts', 'channel-finder']);
    expect(getEntry(rail, 'ariel')).toBeNull();
  });

  test('a removed row can be re-added (remove ≠ forget)', () => {
    removeEntry(rail, 'ariel');
    addEntry(rail, { id: 'ariel', label: 'Ariel' });
    expect(entryIds(rail)).toEqual(['artifacts', 'channel-finder', 'ariel']);
  });

  test('is a no-op for an unknown id', () => {
    expect(() => removeEntry(rail, 'nope')).not.toThrow();
    expect(entryIds(rail)).toEqual(['artifacts', 'ariel', 'channel-finder']);
  });
});

/**
 * placeEntry is the client half of the server's rail-order move: the row
 * moves in the rail order, so it moves within its own group. The Agent row
 * stays first, and `+ Add panel` stays the last row of the last group.
 */
describe('placeEntry', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = pageRail();
    createRail(rail, [
      { id: 'terminal', label: 'Agent' },
      { id: 'artifacts', label: 'Artifacts' },
      { id: 'ariel', label: 'Ariel' },
      { id: 'channel-finder', label: 'Channels' },
      { id: 'lattice', label: 'Lattice' },
    ]);
  });

  /** @returns {string[]} */
  const tools = () => childKeys(group(rail, 'tools'));

  test('moves a row directly above another of its group', () => {
    placeEntry(rail, 'lattice', 'ariel');
    expect(tools()).toEqual(['label:Tools', 'lattice', 'ariel', 'channel-finder', 'panel-add-btn']);
  });

  test('a null beforeId puts the row at the foot of its group, above + Add panel', () => {
    placeEntry(rail, 'ariel', null);
    expect(tools()).toEqual(['label:Tools', 'channel-finder', 'lattice', 'ariel', 'panel-add-btn']);
  });

  test('a beforeId with no row is the foot too', () => {
    placeEntry(rail, 'ariel', 'nope');
    expect(tools()).toEqual(['label:Tools', 'channel-finder', 'lattice', 'ariel', 'panel-add-btn']);
  });

  test('a beforeId in an earlier group puts the row at the head of its own group', () => {
    placeEntry(rail, 'lattice', 'artifacts');
    expect(tools()).toEqual(['label:Tools', 'lattice', 'ariel', 'channel-finder', 'panel-add-btn']);
  });

  test('the Agent row never moves, and no row leaves its group', () => {
    placeEntry(rail, 'terminal', null);
    placeEntry(rail, 'artifacts', 'lattice');
    expect(entryIds(rail)).toEqual(['terminal', 'artifacts', 'ariel', 'channel-finder', 'lattice']);
    expect(childKeys(group(rail, 'workspace'))).toEqual([
      'label:Workspace',
      'sidebar-agent-row',
      'sidebar-artifacts-row',
      'sidebar-runs-row',
    ]);
  });

  test('the rows of a group keep the rail order, rows that join it included', () => {
    placeEntry(rail, 'lattice', 'ariel');
    moveToGroup('channel-finder', 'workspace');
    moveToGroup('lattice', 'workspace');
    expect(childKeys(group(rail, 'workspace'))).toEqual([
      'label:Workspace',
      'sidebar-agent-row',
      'sidebar-artifacts-row',
      'sidebar-runs-row',
      'lattice',
      'channel-finder',
    ]);
  });

  test('an unknown id is a no-op', () => {
    expect(() => placeEntry(rail, 'nope', 'ariel')).not.toThrow();
    expect(entryIds(rail)).toEqual(['terminal', 'artifacts', 'ariel', 'channel-finder', 'lattice']);
  });

  test('the moved row keeps its node and its live state', () => {
    const lattice = row(rail, 'lattice');
    setActive(rail, 'lattice');
    setEntryEnabled(rail, 'lattice', true);

    placeEntry(rail, 'lattice', 'ariel');

    expect(getEntry(rail, 'lattice')).toBe(lattice);
    expect(lattice.classList.contains('active')).toBe(true);
    expect(lattice.classList.contains('disabled')).toBe(false);
  });
});
