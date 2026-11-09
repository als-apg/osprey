// @ts-check
/**
 * Unit tests for the graph finder's input handling (graph-finder-events.js):
 * the background menu of the results area, what keeps the browser's own menu,
 * and the keyboard route to the background menu. The finder is drawn from the
 * real skeleton and a rendered table under a pane, wired with
 * bindFinderEvents against a fake FinderHost:
 *   npx vitest run tests/interfaces/channel_finder/graph-finder-events.test.mjs
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import { bindFinderEvents } from '../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-events.js';
import {
  resultsHtml,
  skeletonHtml,
} from '../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-render.js';
import {
  createFinderState,
  toggleFacet,
  toggleSelection,
} from '../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-state.js';
import { COLUMN_SPECS } from '../../../src/osprey/interfaces/channel_finder/static/js/graph-finder-prefs.js';
import { defaultLayout, visibleColumns } from '/design-system/js/table-columns.js';

const ROWS = [{ fullPv: 'SR01:BPM1:X', device_uri: 'dev:bpm1', device: 'BPM1' }];

/** @type {HTMLElement} */
let pane;
/** @type {() => void} */
let unbind;
/** @type {any} */
let host;

/**
 * Draw the finder and wire its events.
 * @param {{rows?: any[]}} [opts]
 */
function mount({ rows = ROWS } = {}) {
  const finder = createFinderState();
  const layout = defaultLayout(COLUMN_SPECS);
  pane = document.createElement('div');
  pane.innerHTML = `
    <div class="graph-explorer" data-pipeline="graph">
      <div class="graph-explorer-body">${skeletonHtml()}</div>
    </div>`;
  document.body.appendChild(pane);
  const results = { rows, total: rows.length };
  host = {
    finder: () => finder,
    results: () => results,
    card: () => null,
    layout: () => layout,
    setLayout: vi.fn(),
    prefs: () => ({ density: 'comfortable', collapsed: [] }),
    embedded: () => false,
    pane: () => pane,
    part: (/** @type {string} */ id) => pane.querySelector(`#${id}`),
    searchInput: () => null,
    runSearch: vi.fn(async () => {}),
    renderTable: vi.fn(),
    renderRail: vi.fn(),
    commitColumns: vi.fn(),
    toggleDevice: vi.fn(async () => {}),
    openDevice: vi.fn(async () => {}),
    closeCard: vi.fn(),
    copy: vi.fn(async () => {}),
    send: vi.fn(),
    copySelection: vi.fn(),
    turnPage: vi.fn(),
  };
  renderTableInto();
  unbind = bindFinderEvents(pane, host);
}

function renderTableInto() {
  const table = /** @type {HTMLElement} */ (pane.querySelector('#graph-finder-table'));
  table.innerHTML = resultsHtml(
    host.results().rows,
    host.finder(),
    visibleColumns(COLUMN_SPECS, host.layout())
  );
}

/**
 * Right-click an element.
 * @param {Element} el
 * @param {MouseEventInit} [init]
 * @returns {MouseEvent}
 */
function rightClick(el, init = {}) {
  const event = new MouseEvent('contextmenu', {
    bubbles: true,
    cancelable: true,
    button: 2,
    clientX: 10,
    clientY: 10,
    ...init,
  });
  el.dispatchEvent(event);
  return event;
}

/** @returns {HTMLElement|null} the open menu */
function openMenu() {
  return document.querySelector('[role="menu"]');
}

/** @returns {string[]} the open menu's row labels */
function menuLabels() {
  return [...(openMenu()?.querySelectorAll('[role="menuitem"]') ?? [])].map((el) =>
    (el.textContent || '').trim()
  );
}

/** @param {string} label @returns {HTMLElement} */
function menuRow(label) {
  const row = [...(openMenu()?.querySelectorAll('[role="menuitem"]') ?? [])].find(
    (el) => (el.textContent || '').trim() === label
  );
  if (!(row instanceof HTMLElement)) throw new Error(`no menu row ${label}`);
  return row;
}

const results = () => /** @type {HTMLElement} */ (pane.querySelector('.finder-results'));

beforeEach(() => mount());

afterEach(() => {
  document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }));
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  unbind?.();
  pane?.remove();
  document.body.innerHTML = '';
  window.getSelection?.()?.removeAllRanges();
});

describe('the background menu', () => {
  test('right-clicking the results area opens the three background rows', () => {
    const event = rightClick(results());
    expect(event.defaultPrevented).toBe(true);
    expect(menuLabels()).toEqual(['Clear all filters', 'Clear selection', 'Choose columns…']);
  });

  test('the empty area around the table and the pane body open it too', () => {
    for (const sel of [
      '.finder-table',
      '.result-wrap',
      '.graph-explorer-body',
      '.graph-explorer',
    ]) {
      const el = /** @type {Element} */ (pane.querySelector(sel));
      const event = rightClick(el);
      expect(event.defaultPrevented, sel).toBe(true);
      expect(menuLabels(), sel).toContain('Choose columns…');
      document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    }
  });

  test('no row is styled as dangerous', () => {
    rightClick(results());
    for (const el of openMenu()?.querySelectorAll('[role="menuitem"]') ?? []) {
      expect(el.className).not.toMatch(/danger/);
    }
  });

  test('the clears are disabled with nothing to clear', () => {
    rightClick(results());
    expect(menuRow('Clear all filters').getAttribute('aria-disabled')).toBe('true');
    expect(menuRow('Clear selection').getAttribute('aria-disabled')).toBe('true');
    expect(menuRow('Choose columns…').getAttribute('aria-disabled')).not.toBe('true');
  });

  test("Clear all filters runs the finder's clear and searches again", () => {
    toggleFacet(host.finder(), 'section', 'SR01');
    rightClick(results());
    menuRow('Clear all filters').click();
    expect(host.finder().section.size).toBe(0);
    expect(host.runSearch).toHaveBeenCalled();
  });

  test('Clear selection empties the selection and redraws', () => {
    toggleSelection(host.finder(), 'SR01:BPM1:X');
    rightClick(results());
    menuRow('Clear selection').click();
    expect(host.finder().selected.size).toBe(0);
    expect(host.renderTable).toHaveBeenCalled();
  });

  test('Choose columns… opens the column checklist', async () => {
    rightClick(results());
    menuRow('Choose columns…').click();
    await new Promise((r) => setTimeout(r, 0));
    expect(openMenu()?.getAttribute('aria-label')).toBe('Columns');
  });

  test('a row keeps its own menu', () => {
    const cell = /** @type {Element} */ (pane.querySelector('.result-table tbody tr[data-pv] td'));
    rightClick(cell);
    expect(openMenu()?.getAttribute('aria-label')).toBe('SR01:BPM1:X actions');
  });
});

describe('what keeps the browser menu', () => {
  test('Shift+right-click on the background', () => {
    const event = rightClick(results(), { shiftKey: true });
    expect(event.defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });

  test('Shift+right-click on a row', () => {
    const row = /** @type {Element} */ (pane.querySelector('.result-table tbody tr[data-pv]'));
    const event = rightClick(row, { shiftKey: true });
    expect(event.defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });

  test('a link in the results area', () => {
    const caution = /** @type {HTMLElement} */ (pane.querySelector('#graph-finder-caution'));
    caution.innerHTML = '<a href="https://example.org/docs">docs</a>';
    const event = rightClick(/** @type {Element} */ (caution.querySelector('a')));
    expect(event.defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });

  test('a form control in the results area', () => {
    const chips = /** @type {HTMLElement} */ (pane.querySelector('#graph-finder-chips'));
    chips.innerHTML = '<select><option>a</option></select><input type="text">';
    for (const sel of ['select', 'input']) {
      const event = rightClick(/** @type {Element} */ (chips.querySelector(sel)));
      expect(event.defaultPrevented, sel).toBe(false);
    }
    expect(openMenu()).toBeNull();
  });

  test('the inspector body', () => {
    const card = /** @type {HTMLElement} */ (pane.querySelector('.finder-inspector'));
    card.hidden = false;
    card.innerHTML =
      '<div class="device-card"><div class="device-card-body"><p>text</p></div></div>';
    const event = rightClick(/** @type {Element} */ (card.querySelector('p')));
    expect(event.defaultPrevented).toBe(false);
    expect(rightClick(card).defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });

  test('the background while text is selected', () => {
    const caution = /** @type {HTMLElement} */ (pane.querySelector('#graph-finder-caution'));
    caution.textContent = 'some caution text';
    const range = document.createRange();
    range.selectNodeContents(caution);
    window.getSelection()?.addRange(range);
    const event = rightClick(results());
    expect(event.defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });

  test('an element outside the bound pane (other CF views never bind)', () => {
    const other = document.createElement('div');
    other.className = 'finder-results';
    document.body.appendChild(other);
    const event = rightClick(other);
    expect(event.defaultPrevented).toBe(false);
    expect(openMenu()).toBeNull();
  });
});

describe('the keyboard route', () => {
  test('a background click makes the results container focusable and focuses it', () => {
    results().dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(results().getAttribute('tabindex')).toBe('-1');
    expect(document.activeElement).toBe(results());
  });

  test('a click on a row does not focus the results container', () => {
    const cell = /** @type {HTMLElement} */ (
      pane.querySelector('.result-table tbody tr[data-pv] td')
    );
    cell.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(document.activeElement).not.toBe(results());
  });

  test('Shift+F10 on the focused results container opens the background menu', () => {
    results().dispatchEvent(new MouseEvent('click', { bubbles: true }));
    const event = new KeyboardEvent('keydown', {
      key: 'F10',
      shiftKey: true,
      bubbles: true,
      cancelable: true,
    });
    results().dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(menuLabels()).toEqual(['Clear all filters', 'Clear selection', 'Choose columns…']);
  });

  test('the menu key on the focused results container opens it too', () => {
    results().focus();
    results().dispatchEvent(
      new KeyboardEvent('keydown', { key: 'ContextMenu', bubbles: true, cancelable: true })
    );
    expect(menuLabels()).toContain('Clear selection');
  });
});
