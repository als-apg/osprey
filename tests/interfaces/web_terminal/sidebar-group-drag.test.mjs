// @ts-check
/**
 * Dragging a whole sidebar group by its heading (sidebar-group-drag.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-group-drag.test.mjs
 *
 * A heading dragged along the rail moves its group, heading and rows
 * together, among the other groups; the order is stored in the groups model
 * (`osprey-sidebar-groups`), so a reload draws it again. The headless Sessions
 * group is neither dragged nor a target. A folded group drags as its heading.
 * A row drag is untouched: it still reorders rows and moves them between
 * groups, a drop on a heading included, and never reorders the groups.
 *
 * The rail is the real one (panel-rail.js createRail) over the real model,
 * kept in this test's localStorage and cleared before every test. happy-dom
 * lays nothing out, so heights are stubbed (see layOut).
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { byId, qs } from '../_support/dom.mjs';
import { createRail } from '../../../src/osprey/interfaces/web_terminal/static/js/panel-rail.js';
import {
  GROUPS_STORAGE_KEY,
  addGroup,
  groupOf,
  loadGroups,
} from '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-groups.js';
import { initSidebarFold } from '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-fold.js';
import { GROUP_DRAG_MIME } from '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-group-drag.js';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const PANELS = [
  { id: 'terminal', label: 'Agent' },
  { id: 'events', label: 'Events' },
  { id: 'ariel', label: 'Ariel' },
  { id: 'channel-finder', label: 'Channels' },
  { id: 'lattice', label: 'Lattice' },
];

/** @type {HTMLElement} */
let rail;
/** @type {[string, string | null][]} */
let rowMoves;

/** The group ids of the model in force, in sidebar order. */
const modelOrder = () => loadGroups().groups.map((group) => group.id);

/** The group ids as the rail shows them, top to bottom. @param {HTMLElement} [el] */
const railOrder = (el = rail) =>
  [...el.querySelectorAll(':scope > .sidebar-group')].map((g) => g.getAttribute('data-group'));

/** @param {string} id @returns {HTMLElement} */
const group = (id) => qs(rail, `:scope > .sidebar-group[data-group="${id}"]`);

/** @param {string} id @returns {HTMLElement} */
const heading = (id) => qs(group(id), ':scope > .sidebar-group-head');

/** @param {string} id @returns {HTMLElement} */
const row = (id) => qs(rail, `[data-panel-id="${id}"]`);

/**
 * Stack every group top to bottom: a heading in view is 24 px, every other
 * child in view 40 px, and a hidden child (or one a fold hides) nothing. Each
 * group's rect spans its children. With the default groups that puts
 * Sessions (the Agent row) at 0–40, Apps (heading and three rows) at 40–184
 * and System (heading and Events) at 184–248.
 */
function layOut() {
  let top = 0;
  for (const g of rail.querySelectorAll(':scope > .sidebar-group')) {
    const groupTop = top;
    const folded = g.hasAttribute('data-folded');
    for (const el of g.children) {
      const node = /** @type {HTMLElement} */ (el);
      const isHead = node.classList.contains('sidebar-group-head');
      if (node.hidden || (folded && !isHead)) {
        node.getBoundingClientRect = () => /** @type {DOMRect} */ ({ top: 0, height: 0 });
        continue;
      }
      const rect = /** @type {DOMRect} */ ({ top, height: isHead ? 24 : 40, left: 0, width: 200 });
      node.getBoundingClientRect = () => rect;
      top += rect.height;
    }
    const rect = /** @type {DOMRect} */ ({
      top: groupTop,
      height: top - groupTop,
      left: 0,
      width: 200,
    });
    /** @type {HTMLElement} */ (g).getBoundingClientRect = () => rect;
  }
}

/**
 * A drag event happy-dom can dispatch (it has no DragEvent).
 * @param {string} type
 * @param {number} [y] the pointer's height over the sidebar
 */
function dragEvent(type, y = 0) {
  /** @type {Record<string, string>} */
  const data = {};
  const ev = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(ev, {
    clientX: 100,
    clientY: y,
    dataTransfer: {
      dropEffect: '',
      effectAllowed: '',
      setData: (/** @type {string} */ k, /** @type {string} */ v) => {
        data[k] = v;
      },
      getData: (/** @type {string} */ k) => data[k] ?? '',
      data,
    },
  });
  return /** @type {Event & { dataTransfer: { data: Record<string, string> } }} */ (ev);
}

/**
 * Drag a heading and drop it at height `y`, as a browser delivers it.
 * @param {string} id @param {number} y
 * @returns {{ start: Event, over: Event }}
 */
function dragGroup(id, y) {
  const start = dragEvent('dragstart');
  heading(id).dispatchEvent(start);
  const over = dragEvent('dragover', y);
  rail.dispatchEvent(over);
  rail.dispatchEvent(dragEvent('drop', y));
  heading(id).dispatchEvent(dragEvent('dragend'));
  return { start, over };
}

/** The groups carrying the drop line, as [id, 'before' | 'after']. */
function marked() {
  return [...rail.querySelectorAll(':scope > .sidebar-group')]
    .filter((g) => g.matches('.drop-before, .drop-after'))
    .map((g) => [g.getAttribute('data-group'), g.matches('.drop-before') ? 'before' : 'after']);
}

/** Draw a fresh rail inside the sidebar, as the page has it. */
function mount() {
  document.body.innerHTML = '<nav class="sidebar"><div id="panel-rail"></div></nav>';
  rail = byId('panel-rail');
  rowMoves = [];
  createRail(rail, PANELS, {
    onDragStart: () => true,
    onReorder: (id, before) => rowMoves.push([id, before]),
  });
  layOut();
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute('data-sidebar');
  mount();
});

afterEach(() => {
  document.documentElement.removeAttribute('data-sidebar');
  vi.restoreAllMocks();
});

describe('dragging a group by its heading', () => {
  test('a group dropped above another moves there, heading and rows together', () => {
    const { start, over } = dragGroup('system', 50); // upper half of Apps (40–184)
    expect(start.defaultPrevented).toBe(false);
    expect(over.defaultPrevented).toBe(true);
    expect(modelOrder()).toEqual(['workspace', 'system', 'tools']);
    expect(railOrder()).toEqual(['workspace', 'system', 'tools']);
    expect(group('system').contains(row('events'))).toBe(true);
    expect(group('tools').contains(row('ariel'))).toBe(true);
    expect(groupOf('events')).toBe('system');
  });

  test('a group dropped past the last one goes last', () => {
    dragGroup('tools', 240); // below the middle of System (184–248)
    expect(modelOrder()).toEqual(['workspace', 'system', 'tools']);
    expect(railOrder()).toEqual(['workspace', 'system', 'tools']);
  });

  test("the drag carries its own type, dims the group and draws the rows' drop line", () => {
    const start = dragEvent('dragstart');
    heading('system').dispatchEvent(start);
    expect(start.dataTransfer.data[GROUP_DRAG_MIME]).toBe('system');
    expect(group('system').classList.contains('dragging')).toBe(true);

    rail.dispatchEvent(dragEvent('dragover', 50));
    expect(marked()).toEqual([['tools', 'before']]);

    heading('system').dispatchEvent(dragEvent('dragend'));
    expect(group('system').classList.contains('dragging')).toBe(false);
    expect(marked()).toEqual([]);
    expect(modelOrder()).toEqual(['workspace', 'tools', 'system']);
  });

  test('a drop at the end draws the line under the last group', () => {
    heading('tools').dispatchEvent(dragEvent('dragstart'));
    rail.dispatchEvent(dragEvent('dragover', 240));
    expect(marked()).toEqual([['system', 'after']]);
  });

  test('a drop that keeps the order draws no line and changes nothing', () => {
    const before = localStorage.getItem(GROUPS_STORAGE_KEY);
    heading('tools').dispatchEvent(dragEvent('dragstart'));
    rail.dispatchEvent(dragEvent('dragover', 60));
    expect(marked()).toEqual([]);
    rail.dispatchEvent(dragEvent('drop', 60));
    expect(modelOrder()).toEqual(['workspace', 'tools', 'system']);
    expect(localStorage.getItem(GROUPS_STORAGE_KEY)).toBe(before);
  });

  test('nothing lands above Sessions: a drop over it puts the group first after it', () => {
    dragGroup('system', 5);
    expect(modelOrder()).toEqual(['workspace', 'system', 'tools']);
  });

  test('the Sessions heading never drags', () => {
    const start = dragEvent('dragstart');
    heading('workspace').dispatchEvent(start);
    expect(start.defaultPrevented).toBe(true);
    expect(group('workspace').classList.contains('dragging')).toBe(false);
  });

  test('with the sidebar collapsed to icons, no heading drags', () => {
    document.documentElement.setAttribute('data-sidebar', 'collapsed');
    const start = dragEvent('dragstart');
    heading('tools').dispatchEvent(start);
    expect(start.defaultPrevented).toBe(true);
    rail.dispatchEvent(dragEvent('drop', 240));
    expect(modelOrder()).toEqual(['workspace', 'tools', 'system']);
  });

  test("an operator's own group drags among the built-in ones", () => {
    const id = addGroup('Beamline');
    layOut(); // Beamline (heading and empty line) at 248–312
    dragGroup(id, 50);
    expect(modelOrder()).toEqual(['workspace', id, 'tools', 'system']);
    expect(railOrder()).toEqual(['workspace', id, 'tools', 'system']);
  });
});

describe('a folded group', () => {
  test('drags as its heading and stays folded with its rows', () => {
    initSidebarFold(rail);
    heading('tools').dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true }));
    expect(group('tools').hasAttribute('data-folded')).toBe(true);
    layOut(); // Apps folded: its heading alone at 40–64, System at 64–128

    dragGroup('tools', 120); // below the middle of System
    expect(modelOrder()).toEqual(['workspace', 'system', 'tools']);
    expect(group('tools').hasAttribute('data-folded')).toBe(true);
    for (const id of ['ariel', 'channel-finder', 'lattice']) {
      expect(group('tools').contains(row(id))).toBe(true);
    }
  });
});

describe('a row drag is untouched', () => {
  /** @param {string} id */
  const grabRow = (id) => row(id).dispatchEvent(dragEvent('dragstart'));

  test('a row still reorders its group, and the groups keep their order', () => {
    grabRow('lattice');
    rail.dispatchEvent(dragEvent('dragover', 70)); // upper half of ariel (64–104)
    expect(marked()).toEqual([]);
    rail.dispatchEvent(dragEvent('drop', 70));
    row('lattice').dispatchEvent(dragEvent('dragend'));
    expect(rowMoves).toEqual([['lattice', 'ariel']]);
    expect(modelOrder()).toEqual(['workspace', 'tools', 'system']);
  });

  test('a row dropped on a heading still lands at the head of that group', () => {
    grabRow('ariel');
    rail.dispatchEvent(dragEvent('dragover', 190)); // on the System heading (184–208)
    rail.dispatchEvent(dragEvent('drop', 190));
    row('ariel').dispatchEvent(dragEvent('dragend'));
    expect(groupOf('ariel')).toBe('system');
    expect(rowMoves).toEqual([['ariel', 'events']]);
    expect(modelOrder()).toEqual(['workspace', 'tools', 'system']);
  });

  test('a group drag never moves a row between groups', () => {
    dragGroup('system', 50);
    expect(rowMoves).toEqual([]);
    expect(groupOf('ariel')).toBe('tools');
  });
});

describe('persistence', () => {
  test('the order a drag made is stored and drawn again after a reload', async () => {
    dragGroup('system', 50);
    const stored = JSON.parse(localStorage.getItem(GROUPS_STORAGE_KEY) ?? 'null');
    expect(stored.groups.map((/** @type {{id: string}} */ g) => g.id)).toEqual([
      'workspace',
      'system',
      'tools',
    ]);

    // A reload: fresh modules over the same storage, and a fresh rail.
    vi.resetModules();
    const fresh = await import(`${JS}/panel-rail.js`);
    document.body.innerHTML = '<nav class="sidebar"><div id="panel-rail"></div></nav>';
    const again = byId('panel-rail');
    fresh.createRail(again, PANELS);
    expect(railOrder(again)).toEqual(['workspace', 'system', 'tools']);
  });
});
