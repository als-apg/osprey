/**
 * Unit tests for the sidebar's panel rows (DOM renderer).
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-rail.test.mjs
 *
 * panel-rail.js is a pure DOM module: it places the sidebar's panel rows in
 * two groups (Workspace and Tools) and exposes imperative mutators (active /
 * enabled / attention / non-destructive append, place & remove) that
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
 * the shipped index.html (page-owned rows adopted, never rebuilt).
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
  setEntryReachable,
} from '../../../src/osprey/interfaces/web_terminal/static/js/panel-rail.js';
import { icon } from '/design-system/js/icons.js';
import { FACILITY_ZONE, VIEWER_ZONE, stampFacilityZone } from '../_support/facility-zone.mjs';

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

/** The direct children of a group, by id / data-panel-id / heading.
 *  @param {HTMLElement} groupEl @returns {string[]} */
function childKeys(groupEl) {
  return [...groupEl.children].map((el) =>
    el.classList.contains('osprey-section-label')
      ? `label:${el.textContent}`
      : el.id || String(el.getAttribute('data-panel-id'))
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

  test('marks the container as a list of two labelled groups', () => {
    createRail(rail, PANELS);
    expect(rail.getAttribute('role')).toBe('list');
    const groups = [...rail.children];
    expect(groups.map((g) => g.getAttribute('data-group'))).toEqual(['workspace', 'tools']);
    for (const g of groups) {
      expect(g.classList.contains('sidebar-group')).toBe(true);
      expect(g.getAttribute('role')).toBe('listitem');
      const label = /** @type {HTMLElement} */ (g.firstElementChild);
      expect(label.classList.contains('osprey-section-label')).toBe(true);
      expect(g.getAttribute('aria-labelledby')).toBe(label.id);
    }
    expect(groups.map((g) => g.firstElementChild?.textContent)).toEqual(['Workspace', 'Tools']);
  });

  test('a Workspace panel sits under Workspace, every other one under Tools', () => {
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
    expect(group(rail, 'workspace').id).toBe('sidebar-workspace');
  });

  test('Workspace keeps its fixed order and Tools follows the rail order, then + Add panel', () => {
    createRail(rail, ALL);
    expect(childKeys(group(rail, 'workspace'))).toEqual([
      'label:Workspace',
      'sidebar-agent-row',
      'sidebar-artifacts-row',
      'sidebar-runs-row',
      'events',
    ]);
    expect(childKeys(group(rail, 'tools'))).toEqual([
      'label:Tools',
      'ariel',
      'lattice',
      'panel-add-btn',
    ]);
    expect(entryIds(rail)).toEqual(['terminal', 'artifacts', 'events', 'ariel', 'lattice']);
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

  test('a built Workspace row added at runtime takes its fixed place', () => {
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
 * Reordering moves Tools rows among themselves, vertically. Workspace rows
 * keep the sidebar's fixed order: they are neither dragged nor targets, and a
 * Tools row the caller marks fixed (`reorderable` false) is neither either.
 */
describe('reorder among the Tools rows (onReorder)', () => {
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
  beforeEach(() => {
    rail = freshRail();
    moves = [];
    createRail(rail, ROW, {
      onDragStart: () => true,
      onReorder: (id, before) => moves.push([id, before]),
      reorderable: (id) => id !== 'okf',
    });
    // happy-dom has no layout: stack the rows 40px apart, top to bottom —
    // terminal 0, artifacts 40, ariel 80, channel-finder 120, lattice 160,
    // okf 200. Every row shares one column, so x never decides anything.
    ROW.forEach(({ id }, i) => {
      row(rail, id).getBoundingClientRect = () =>
        /** @type {DOMRect} */ ({ top: i * 40, height: 40, left: 0, width: 200 });
    });
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

  test('dropping above a Tools row moves the dragged one above it', () => {
    grab('lattice');
    const over = pointerEvent('dragover', 90); // upper half of ariel (80–120)
    rail.dispatchEvent(over);
    expect(over.defaultPrevented).toBe(true);
    expect(marked()).toEqual([['ariel', 'before']]);

    rail.dispatchEvent(pointerEvent('drop', 90));
    expect(moves).toEqual([['lattice', 'ariel']]);
    expect(marked()).toEqual([]);
  });

  test('dropping past the last movable row moves the dragged one to the foot', () => {
    grab('ariel');
    rail.dispatchEvent(pointerEvent('dragover', 230)); // over the fixed okf row
    expect(marked()).toEqual([['lattice', 'after']]);
    rail.dispatchEvent(pointerEvent('drop', 230));
    expect(moves).toEqual([['ariel', null]]);
  });

  test('a Workspace row is never a target: above the Agent row lands at the head of Tools', () => {
    grab('channel-finder');
    rail.dispatchEvent(pointerEvent('dragover', 5));
    expect(marked()).toEqual([['ariel', 'before']]);
    rail.dispatchEvent(pointerEvent('drop', 5));
    expect(moves).toEqual([['channel-finder', 'ariel']]);
  });

  test('a drop that leaves the row where it is reports no move', () => {
    grab('channel-finder');
    rail.dispatchEvent(pointerEvent('dragover', 125)); // just below its own top
    expect(marked()).toEqual([]);
    rail.dispatchEvent(pointerEvent('drop', 125));
    expect(moves).toEqual([]);
  });

  test.each(['terminal', 'artifacts'])('the Workspace row %s cannot be reordered', (id) => {
    grab(id);
    const over = pointerEvent('dragover', 150);
    rail.dispatchEvent(over);
    expect(over.defaultPrevented).toBe(false);
    rail.dispatchEvent(pointerEvent('drop', 150));
    expect(moves).toEqual([]);
  });

  test('a Tools row the caller fixes cannot be reordered', () => {
    grab('okf');
    const over = pointerEvent('dragover', 90);
    rail.dispatchEvent(over);
    expect(over.defaultPrevented).toBe(false);
    rail.dispatchEvent(pointerEvent('drop', 90));
    expect(moves).toEqual([]);
  });

  test('a drag that did not start on a row is left alone', () => {
    const over = pointerEvent('dragover', 90);
    rail.dispatchEvent(over);
    expect(over.defaultPrevented).toBe(false);
    rail.dispatchEvent(pointerEvent('drop', 90));
    expect(moves).toEqual([]);
  });

  test('the drop reads the pointer height only', () => {
    grab('lattice');
    const ev = pointerEvent('drop', 90);
    Object.assign(ev, { clientX: 9999 });
    rail.dispatchEvent(ev);
    expect(moves).toEqual([['lattice', 'ariel']]);
  });

  test('dragend clears the insertion line', () => {
    grab('lattice');
    rail.dispatchEvent(pointerEvent('dragover', 90));
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
    ROW.forEach(({ id }, i) => {
      row(rail, id).getBoundingClientRect = () =>
        /** @type {DOMRect} */ ({ top: i * 40, height: 40, left: 0, width: 200 });
    });
    grab('okf');
    rail.dispatchEvent(pointerEvent('drop', 90));
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

  test('a Workspace row added at runtime takes its fixed place, not the foot', () => {
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
 * placeEntry moves Tools rows only. Workspace rows keep the sidebar's fixed
 * order, and `+ Add panel` stays the last row of Tools.
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

  test('moves a Tools row directly above another', () => {
    placeEntry(rail, 'lattice', 'ariel');
    expect(tools()).toEqual(['label:Tools', 'lattice', 'ariel', 'channel-finder', 'panel-add-btn']);
  });

  test('a null beforeId puts the row at the foot of Tools, above + Add panel', () => {
    placeEntry(rail, 'ariel', null);
    expect(tools()).toEqual(['label:Tools', 'channel-finder', 'lattice', 'ariel', 'panel-add-btn']);
  });

  test('a beforeId with no row is the foot of Tools too', () => {
    placeEntry(rail, 'ariel', 'nope');
    expect(tools()).toEqual(['label:Tools', 'channel-finder', 'lattice', 'ariel', 'panel-add-btn']);
  });

  test('a beforeId in Workspace puts the row at the head of Tools', () => {
    placeEntry(rail, 'lattice', 'artifacts');
    expect(tools()).toEqual(['label:Tools', 'lattice', 'ariel', 'channel-finder', 'panel-add-btn']);
  });

  test('a Workspace row never moves', () => {
    placeEntry(rail, 'terminal', null);
    placeEntry(rail, 'artifacts', 'ariel');
    expect(entryIds(rail)).toEqual(['terminal', 'artifacts', 'ariel', 'channel-finder', 'lattice']);
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

describe('setActive', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('marks exactly one row active and current', () => {
    setActive(rail, 'ariel');
    expect(row(rail, 'ariel').classList.contains('active')).toBe(true);
    expect(row(rail, 'ariel').getAttribute('aria-current')).toBe('page');
    expect(row(rail, 'artifacts').classList.contains('active')).toBe(false);
    expect(row(rail, 'artifacts').hasAttribute('aria-current')).toBe(false);
  });

  test('switching active clears the previous one', () => {
    setActive(rail, 'artifacts');
    setActive(rail, 'channel-finder');
    expect(row(rail, 'artifacts').classList.contains('active')).toBe(false);
    expect(row(rail, 'artifacts').hasAttribute('aria-current')).toBe(false);
    expect(row(rail, 'channel-finder').classList.contains('active')).toBe(true);
  });

  test('a null / unknown id clears active on every row', () => {
    setActive(rail, 'artifacts');
    setActive(rail, null);
    expect(rail.querySelectorAll('.panel-rail-button.active').length).toBe(0);
    expect(rail.querySelectorAll('[aria-current]').length).toBe(0);
  });
});

describe('setEntryEnabled', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('setEntryEnabled toggles the disabled class', () => {
    setEntryEnabled(rail, 'artifacts', true);
    expect(row(rail, 'artifacts').classList.contains('disabled')).toBe(false);
    setEntryEnabled(rail, 'artifacts', false);
    expect(row(rail, 'artifacts').classList.contains('disabled')).toBe(true);
  });

  test('is a no-op for an unknown id', () => {
    expect(() => setEntryEnabled(rail, 'nope', true)).not.toThrow();
  });
});

describe('setEntryAttention', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('on sets the persistent attention class and fires the transient flash', () => {
    expect(setEntryAttention(rail, 'ariel', true)).toBe(true);
    const entry = row(rail, 'ariel');
    expect(entry.classList.contains('agent-attention')).toBe(true);
    expect(entry.classList.contains('agent-flash')).toBe(true);
  });

  test('off clears the attention without force-stopping an in-flight flash', () => {
    setEntryAttention(rail, 'ariel', true);
    expect(setEntryAttention(rail, 'ariel', false)).toBe(true);
    const entry = row(rail, 'ariel');
    expect(entry.classList.contains('agent-attention')).toBe(false);
    // No animationend has fired in happy-dom, so the flash class must survive.
    expect(entry.classList.contains('agent-flash')).toBe(true);
  });

  test('returns false, scrolls nothing and does not throw for an unknown id', () => {
    const scrollSpy = vi.spyOn(Element.prototype, 'scrollIntoView').mockImplementation(() => {});
    expect(() => setEntryAttention(rail, 'nope', true)).not.toThrow();
    expect(setEntryAttention(rail, 'nope', true)).toBe(false);
    expect(scrollSpy).not.toHaveBeenCalled();
    scrollSpy.mockRestore();
  });

  test('on draws one hidden-from-AT dot after the label; off removes it', () => {
    const entry = row(rail, 'ariel');
    const childrenBefore = [...entry.children];

    setEntryAttention(rail, 'ariel', true);
    setEntryAttention(rail, 'ariel', true);
    const dots = entry.querySelectorAll('[data-attention-badge]');
    expect(dots.length).toBe(1);
    const dot = /** @type {HTMLElement} */ (dots[0]);
    expect(dot.classList.contains('osprey-nav-badge')).toBe(true);
    expect(dot.classList.contains('sidebar-attention')).toBe(true);
    expect(dot.getAttribute('aria-hidden')).toBe('true');
    expect(dot.previousElementSibling?.classList.contains('osprey-nav-label')).toBe(true);

    setEntryAttention(rail, 'ariel', false);
    expect([...entry.children]).toEqual(childrenBefore);
  });

  test('without a time, a badge and its clear leave the row attributes as they were', () => {
    const entry = row(rail, 'ariel');
    const attrsBefore = entry.getAttributeNames().sort();

    setEntryAttention(rail, 'ariel', true);
    setEntryAttention(rail, 'ariel', false);

    expect(entry.getAttributeNames().sort()).toEqual(attrsBefore);
  });

  test('a badged-then-cleared row leaves no tooltip stash behind', () => {
    const entry = row(rail, 'ariel');
    setEntryAttention(rail, 'ariel', true, 1_755_000_000);
    expect(entry.hasAttribute('data-title-base')).toBe(true);

    setEntryAttention(rail, 'ariel', false);
    expect(entry.hasAttribute('data-title-base')).toBe(false);
  });
});

/**
 * A sidebar taller than its viewport can put a row off-screen, where a badge
 * reports nothing at all. Setting the badge must surface the row; clearing it
 * must not move the sidebar under the operator.
 *
 * happy-dom implements scrollIntoView as a no-op on Element.prototype, so the
 * spy below observes real calls rather than installing a missing method.
 */
describe('setEntryAttention scrolls a badged row into view', () => {
  /** @type {HTMLElement} */
  let rail;
  /** @type {import('vitest').MockInstance} */
  let scrollSpy;

  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
    scrollSpy = vi.spyOn(Element.prototype, 'scrollIntoView').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('setting the badge surfaces the row without disturbing a visible sidebar', () => {
    setEntryAttention(rail, 'ariel', true);

    expect(scrollSpy).toHaveBeenCalledTimes(1);
    // `nearest` is the whole point: a row already in view must not scroll.
    expect(scrollSpy).toHaveBeenCalledWith({ block: 'nearest' });
    expect(scrollSpy.mock.instances[0]).toBe(getEntry(rail, 'ariel'));
  });

  test('clearing the badge does not scroll', () => {
    setEntryAttention(rail, 'ariel', true);
    scrollSpy.mockClear();

    setEntryAttention(rail, 'ariel', false);
    expect(scrollSpy).not.toHaveBeenCalled();
  });
});

/**
 * The badge says a panel was touched; the tooltip says WHEN. The time comes
 * from the event's own server `ts` (epoch seconds, as the agent_activity SSE
 * frames carry it) — never from the client clock, which would report when the
 * browser rendered rather than when the agent acted.
 */
describe('setEntryAttention tooltip time', () => {
  /** @type {HTMLElement} */
  let rail;

  // Two fixed server timestamps an hour apart. Rendering is asserted against
  // an Intl computation in the stamped facility zone rather than a literal so
  // the suite is not hostage to the runner's timezone or locale; the FORMAT is
  // pinned separately. FACILITY_ZONE reads a different clock from the runner,
  // so the time names its zone.
  const TS = 1_755_000_000;
  const TS_LATER = TS + 3600;

  /** @param {number} ts @returns {string} */
  const expectedTime = (ts) =>
    new Intl.DateTimeFormat(undefined, {
      hour: 'numeric',
      minute: '2-digit',
      timeZone: FACILITY_ZONE,
      timeZoneName: 'short',
    }).format(ts * 1000);

  beforeEach(() => {
    stampFacilityZone(FACILITY_ZONE);
    rail = freshRail();
    createRail(rail, PANELS);
  });

  afterEach(() => stampFacilityZone(null));

  test('a viewer already on the facility clock reads the time without a zone name', () => {
    stampFacilityZone(VIEWER_ZONE);
    setEntryAttention(rail, 'ariel', true, TS);
    const plain = new Intl.DateTimeFormat(undefined, {
      hour: 'numeric',
      minute: '2-digit',
      timeZone: VIEWER_ZONE,
    }).format(TS * 1000);
    expect(row(rail, 'ariel').title).toBe(`Ariel · agent touched ${plain}`);
  });

  test('a badge with a server ts appends the touch time to the tooltip', () => {
    setEntryAttention(rail, 'ariel', true, TS);
    expect(row(rail, 'ariel').title).toBe(`Ariel · agent touched ${expectedTime(TS)}`);
  });

  test('the rendered time is hour and minute only — no seconds', () => {
    setEntryAttention(rail, 'ariel', true, TS);
    const suffix = String(row(rail, 'ariel').title).split('· agent touched ')[1];
    expect(suffix).toMatch(/\d{1,2}:\d{2}/);
    expect((suffix.match(/:/g) ?? []).length).toBe(1);
  });

  test('a second event replaces the time rather than appending a second suffix', () => {
    setEntryAttention(rail, 'ariel', true, TS);
    setEntryAttention(rail, 'ariel', true, TS_LATER);

    const title = String(row(rail, 'ariel').title);
    expect(title).toBe(`Ariel · agent touched ${expectedTime(TS_LATER)}`);
    expect((title.match(/agent touched/g) ?? []).length).toBe(1);
  });

  test('clearing the badge restores the base tooltip exactly', () => {
    setEntryAttention(rail, 'ariel', true, TS);
    setEntryAttention(rail, 'ariel', false);

    expect(row(rail, 'ariel').title).toBe('Ariel');
  });

  test('a later badge with no ts drops the stale time instead of keeping it', () => {
    setEntryAttention(rail, 'ariel', true, TS);
    setEntryAttention(rail, 'ariel', true);

    expect(row(rail, 'ariel').title).toBe('Ariel');
  });

  test('no ts leaves the tooltip untouched (the pre-existing 3-arg contract)', () => {
    setEntryAttention(rail, 'ariel', true);
    expect(row(rail, 'ariel').title).toBe('Ariel');
    expect(row(rail, 'ariel').classList.contains('agent-attention')).toBe(true);
  });

  test('a non-finite ts is refused rather than rendered as "Invalid Date"', () => {
    for (const bad of [NaN, Infinity]) {
      setEntryAttention(rail, 'ariel', true, bad);
      expect(row(rail, 'ariel').title).toBe('Ariel');
    }
  });

  test('the badge borrows and returns a hinted tooltip intact', () => {
    // The stash is what the badge restores; a row advertising its menu must
    // get that advertisement back, not just the bare label.
    rail = freshRail();
    createRail(rail, [{ id: 'ariel', label: 'Ariel', hint: 'right-click for actions' }]);

    setEntryAttention(rail, 'ariel', true, TS);
    expect(row(rail, 'ariel').title).toBe(
      `Ariel · right-click for actions · agent touched ${expectedTime(TS)}`
    );

    setEntryAttention(rail, 'ariel', false);
    expect(row(rail, 'ariel').title).toBe('Ariel · right-click for actions');
  });

  test('the suffix never reaches the accessible name', () => {
    // aria-label is the row's identity; a churning timestamp inside it would
    // make the sidebar re-announce panels to a screen reader on every event.
    setEntryAttention(rail, 'ariel', true, TS);
    expect(row(rail, 'ariel').getAttribute('aria-label')).toBe('Ariel');
  });
});

/**
 * Coarse reachability: a row whose backend answered before but has stopped
 * is dimmed without going inert, and its tooltip says since when. The poll
 * logic that decides WHEN is panel-lifecycle's (pinned in panel-manager.test.mjs).
 */
describe('setEntryReachable', () => {
  /** @type {HTMLElement} */
  let rail;
  const SINCE = new Date(2026, 0, 5, 9, 12).getTime();
  // On the facility clock, which reads differently from the runner's, so the
  // time names its zone.
  const since = new Intl.DateTimeFormat(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    timeZone: FACILITY_ZONE,
    timeZoneName: 'short',
  }).format(SINCE);

  beforeEach(() => {
    stampFacilityZone(FACILITY_ZONE);
    rail = freshRail();
    createRail(rail, [{ id: 'ariel', label: 'Ariel', hint: 'right-click for actions' }]);
  });

  afterEach(() => stampFacilityZone(null));

  test('unreachable dims without disabling, and the tooltip says since when', () => {
    expect(setEntryReachable(rail, 'ariel', false, SINCE)).toBe(true);
    const entry = row(rail, 'ariel');
    expect(entry.classList.contains('unreachable')).toBe(true);
    expect(entry.classList.contains('disabled')).toBe(true); // the cold boot state, untouched
    expect(entry.title).toBe(`not answering since ${since}`);

    setEntryReachable(rail, 'ariel', true);
    expect(entry.classList.contains('unreachable')).toBe(false);
    expect(entry.title).toBe('Ariel · right-click for actions');
    expect(entry.hasAttribute('data-title-base')).toBe(false);
  });

  test('the notice outranks a badge time, and each clears without losing the other', () => {
    const entry = row(rail, 'ariel');
    setEntryAttention(rail, 'ariel', true, 1_755_000_000);
    const badged = entry.title;

    setEntryReachable(rail, 'ariel', false, SINCE);
    expect(entry.title).toBe(`not answering since ${since}`);

    setEntryReachable(rail, 'ariel', true);
    expect(entry.title).toBe(badged);

    setEntryAttention(rail, 'ariel', false);
    expect(entry.title).toBe('Ariel · right-click for actions');
  });

  test('returns false for an unknown id', () => {
    expect(setEntryReachable(rail, 'nope', false, SINCE)).toBe(false);
  });
});

/**
 * A sidecar that failed to start is the one status the sidebar names: its row
 * is dimmed but clickable (the click is the retry), and its tooltip is the
 * server's sentence, verbatim.
 */
describe('setEntryStatus', () => {
  /** @type {HTMLElement} */
  let rail;
  const MESSAGE = 'JUPYTER failed to start: boom';
  const JUPYTER = { id: 'jupyter', label: 'Jupyter', hint: 'right-click for actions' };

  beforeEach(() => {
    rail = freshRail();
  });

  test("a failed row is clickable and carries the server's message as its tooltip", () => {
    const onActivate = vi.fn();
    createRail(rail, [JUPYTER], { onActivate });

    setEntryStatus(rail, 'jupyter', { failed: true, message: MESSAGE });

    const entry = row(rail, 'jupyter');
    expect(entry.classList.contains('failed')).toBe(true);
    expect(entry.classList.contains('disabled')).toBe(false);
    expect(entry.title).toBe(MESSAGE);
    expect(entry.getAttribute('aria-description')).toBe(MESSAGE);
    entry.click();
    expect(onActivate).toHaveBeenCalledWith('jupyter');
  });

  test('clearing the status restores the base tooltip', () => {
    createRail(rail, [JUPYTER]);
    setEntryStatus(rail, 'jupyter', { failed: true, message: MESSAGE });

    setEntryStatus(rail, 'jupyter', { failed: false, message: null });

    const entry = row(rail, 'jupyter');
    expect(entry.classList.contains('failed')).toBe(false);
    expect(entry.title).toBe('Jupyter · right-click for actions');
    expect(entry.hasAttribute('aria-description')).toBe(false);
    expect(entry.hasAttribute('data-title-base')).toBe(false);
  });

  test('a message set under an attention badge survives the badge clearing', () => {
    createRail(rail, [JUPYTER]);
    setEntryAttention(rail, 'jupyter', true, 1_755_000_000);

    setEntryStatus(rail, 'jupyter', { failed: true, message: MESSAGE });
    const entry = row(rail, 'jupyter');
    expect(entry.title.startsWith(`${MESSAGE} · agent touched `)).toBe(true);

    setEntryAttention(rail, 'jupyter', false);
    expect(entry.title).toBe(MESSAGE);

    setEntryStatus(rail, 'jupyter', { failed: false, message: null });
    expect(entry.title).toBe('Jupyter · right-click for actions');
  });
});
