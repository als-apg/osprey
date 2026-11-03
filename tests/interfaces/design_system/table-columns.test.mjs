/**
 * Unit tests for movable, resizable table columns (design_system/static/js/table-columns.js).
 *
 *   npx vitest run tests/interfaces/design_system/table-columns.test.mjs
 *
 * The layout model is pure and tested directly. The gesture controller is
 * driven with synthetic pointer events over a table whose header rects are
 * faked, since happy-dom lays nothing out.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  attachColumnGestures,
  defaultLayout,
  isDefaultLayout,
  measureColumn,
  moveColumn,
  nudgeColumn,
  orderedColumns,
  resizeColumn,
  restoreLayout,
  serializeLayout,
  setColumnHidden,
  visibleColumns,
  MAX_WIDTH,
} from '/design-system/js/table-columns.js';

/** @type {import('/design-system/js/table-columns.js').ColumnSpec[]} */
const SPECS = [
  { key: 'chk', label: '', width: 28, pin: 'start' },
  { key: 'device', label: 'Device', width: 150 },
  { key: 'section', label: 'Sec', width: 60, minWidth: 48 },
  { key: 'address', label: 'Address', width: 190 },
  { key: 'system', label: 'System', width: 70, hidden: true },
  { key: 'description', label: 'Description', width: 260 },
  { key: 'act', label: '', width: 34, pin: 'end' },
];

const keys = (/** @type {{key: string}[]} */ cols) => cols.map((c) => c.key);

describe('layout model', () => {
  test('the default layout keeps pins at their edges and hides what the specs hide', () => {
    const layout = defaultLayout(SPECS);
    expect(layout.order).toEqual(['device', 'section', 'address', 'system', 'description']);
    expect(keys(orderedColumns(SPECS, layout))).toEqual([
      'chk',
      'device',
      'section',
      'address',
      'system',
      'description',
      'act',
    ]);
    expect(keys(visibleColumns(SPECS, layout))).toEqual([
      'chk',
      'device',
      'section',
      'address',
      'description',
      'act',
    ]);
  });

  test('a move lands before or after its target and reports whether it changed anything', () => {
    const layout = defaultLayout(SPECS);
    expect(moveColumn(layout, 'description', 'device', true)).toBe(true);
    expect(layout.order).toEqual(['description', 'device', 'section', 'address', 'system']);
    expect(moveColumn(layout, 'description', 'device', false)).toBe(true);
    expect(layout.order).toEqual(['device', 'description', 'section', 'address', 'system']);
    expect(moveColumn(layout, 'description', 'device', false)).toBe(false);
  });

  test('pinned columns can be neither moved nor targeted', () => {
    const layout = defaultLayout(SPECS);
    expect(moveColumn(layout, 'chk', 'device', false)).toBe(false);
    expect(moveColumn(layout, 'device', 'act', false)).toBe(false);
    expect(layout.order).toEqual(defaultLayout(SPECS).order);
  });

  test('a nudge steps over hidden columns, and stops at either end', () => {
    const layout = defaultLayout(SPECS);
    expect(nudgeColumn(layout, 'address', 1)).toBe(true);
    // system is hidden, so address lands after description, not after system
    expect(keys(visibleColumns(SPECS, layout))).toEqual([
      'chk',
      'device',
      'section',
      'description',
      'address',
      'act',
    ]);
    expect(nudgeColumn(layout, 'address', 1)).toBe(false);
    expect(nudgeColumn(layout, 'device', -1)).toBe(false);
  });

  test('a resize clamps to the column minimum and the shared maximum', () => {
    const layout = defaultLayout(SPECS);
    expect(resizeColumn(SPECS, layout, 'section', 10)).toBe(48);
    expect(resizeColumn(SPECS, layout, 'device', 10)).toBe(40);
    expect(resizeColumn(SPECS, layout, 'device', 5000)).toBe(MAX_WIDTH);
    expect(resizeColumn(SPECS, layout, 'device', 201.6)).toBe(202);
    expect(visibleColumns(SPECS, layout).find((c) => c.key === 'device')?.width).toBe(202);
  });

  test('hiding refuses pins and the last visible movable column', () => {
    const layout = defaultLayout(SPECS);
    expect(setColumnHidden(SPECS, layout, 'chk', true)).toBe(false);
    for (const k of ['device', 'section', 'address'])
      expect(setColumnHidden(SPECS, layout, k, true)).toBe(true);
    expect(setColumnHidden(SPECS, layout, 'description', true)).toBe(false);
    expect(setColumnHidden(SPECS, layout, 'system', false)).toBe(true);
    expect(setColumnHidden(SPECS, layout, 'system', false)).toBe(false);
  });

  test('isDefaultLayout tracks whether a reset has anything to undo', () => {
    const layout = defaultLayout(SPECS);
    expect(isDefaultLayout(SPECS, layout)).toBe(true);
    resizeColumn(SPECS, layout, 'device', 180);
    expect(isDefaultLayout(SPECS, layout)).toBe(false);
  });
});

describe('storage round-trip', () => {
  test('a serialized layout restores to the same arrangement, from an object or a string', () => {
    const layout = defaultLayout(SPECS);
    moveColumn(layout, 'address', 'device', true);
    resizeColumn(SPECS, layout, 'description', 300);
    setColumnHidden(SPECS, layout, 'system', false);
    const stored = serializeLayout(layout);
    expect(restoreLayout(SPECS, stored)).toEqual(layout);
    expect(restoreLayout(SPECS, JSON.stringify(stored))).toEqual(layout);
  });

  test('unreadable storage yields the defaults', () => {
    for (const junk of [null, undefined, 42, 'not json', '{"order": 7}']) {
      const restored = restoreLayout(SPECS, junk);
      expect(restored.order).toEqual(defaultLayout(SPECS).order);
    }
  });

  test('unknown columns are dropped and new ones appear at their default slot', () => {
    // Stored before "address" existed, and naming a column that no longer does.
    const restored = restoreLayout(SPECS, {
      order: ['description', 'retired', 'device', 'section', 'system'],
      widths: { device: 222, retired: 99, chk: 500 },
      hidden: ['retired'],
    });
    expect(restored.order).toEqual(['description', 'device', 'section', 'address', 'system']);
    expect(restored.widths.device).toBe(222);
    expect(restored.widths.chk).toBe(28);
    expect('retired' in restored.widths).toBe(false);
    expect([...restored.hidden]).toEqual([]);
  });

  test('a stored layout with every movable column hidden shows them all again', () => {
    const all = defaultLayout(SPECS).order;
    expect([...restoreLayout(SPECS, { order: all, widths: {}, hidden: all }).hidden]).toEqual([]);
  });

  test('stored widths are clamped like a live resize', () => {
    const restored = restoreLayout(SPECS, {
      order: [],
      widths: { section: 1, device: 9999, address: 'wide' },
      hidden: [],
    });
    expect(restored.widths.section).toBe(48);
    expect(restored.widths.device).toBe(MAX_WIDTH);
    expect(restored.widths.address).toBe(190);
  });
});

describe('gestures', () => {
  /** @type {HTMLElement} */
  let region;
  /** @type {() => void} */
  let detach;
  /** @type {Record<string, any[]>} */
  let calls;

  /** Lay the header out at fixed x positions: device 0-100, section 100-200, address 200-300. */
  function layoutHeads() {
    const xs = { device: 0, section: 100, address: 200 };
    for (const [key, left] of Object.entries(xs)) {
      const th = /** @type {HTMLElement} */ (region.querySelector(`th[data-col="${key}"]`));
      vi.spyOn(th, 'getBoundingClientRect').mockReturnValue(
        /** @type {DOMRect} */ ({
          left,
          right: left + 100,
          width: 100,
          top: 0,
          bottom: 20,
          height: 20,
          x: left,
          y: 0,
        })
      );
    }
  }

  /** @param {string} type @param {Element} target @param {number} x @param {object} [extra] */
  function fire(type, target, x, extra = {}) {
    target.dispatchEvent(
      new PointerEvent(type, {
        bubbles: true,
        cancelable: true,
        clientX: x,
        clientY: 5,
        button: 0,
        pointerId: 1,
        ...extra,
      })
    );
  }

  beforeEach(() => {
    document.body.innerHTML = `
      <div id="region"><table><thead><tr>
        <th data-col="chk" data-pinned><input type="checkbox"></th>
        <th data-col="device">Device<span class="col-resize" data-resize="device"></span></th>
        <th data-col="section">Sec<span class="col-resize" data-resize="section"></span></th>
        <th data-col="address">Address<span class="col-resize" data-resize="address"></span></th>
      </tr></thead><tbody></tbody></table></div>`;
    region = /** @type {HTMLElement} */ (document.getElementById('region'));
    calls = { resizing: [], resize: [], move: [], autosize: [], click: [] };
    detach = attachColumnGestures(region, {
      onResizing: (k, w) => calls.resizing.push([k, w]),
      onResize: (k, w) => calls.resize.push([k, w]),
      onMove: (k, t, b) => calls.move.push([k, t, b]),
      onAutosize: (k) => calls.autosize.push(k),
    });
    region.addEventListener('click', (e) =>
      calls.click.push(/** @type {Element} */ (e.target).closest('th')?.getAttribute('data-col'))
    );
    layoutHeads();
  });

  afterEach(() => detach());

  test('dragging an edge reports live widths and one final width', () => {
    const grip = /** @type {Element} */ (region.querySelector('[data-resize="section"]'));
    fire('pointerdown', grip, 200);
    fire('pointermove', grip, 230);
    fire('pointermove', grip, 260);
    fire('pointerup', grip, 260);
    expect(calls.resizing).toEqual([
      ['section', 130],
      ['section', 160],
    ]);
    expect(calls.resize).toEqual([['section', 160]]);
    expect(calls.move).toEqual([]);
    expect(document.body.classList.contains('col-resizing')).toBe(false);
  });

  test('dragging a header past the threshold moves it to the nearest gap', () => {
    const th = /** @type {Element} */ (region.querySelector('th[data-col="device"]'));
    fire('pointerdown', th, 50);
    fire('pointermove', th, 120);
    expect(document.querySelector('.col-drag-ghost')?.textContent).toBe('Device');
    expect(document.querySelector('.col-drop-line')).not.toBeNull();
    fire('pointermove', th, 280);
    fire('pointerup', th, 280);
    expect(calls.move).toEqual([['device', 'address', false]]);
    expect(document.querySelector('.col-drag-ghost')).toBeNull();
    expect(document.querySelector('.col-drop-line')).toBeNull();
  });

  test('the click that ends a drag is swallowed; a plain click is not', () => {
    const th = /** @type {HTMLElement} */ (region.querySelector('th[data-col="device"]'));
    fire('pointerdown', th, 50);
    fire('pointermove', th, 180);
    fire('pointerup', th, 180);
    th.click();
    expect(calls.click).toEqual([]);

    fire('pointerdown', th, 50);
    fire('pointerup', th, 51);
    th.click();
    expect(calls.click).toEqual(['device']);
  });

  test('a press below the threshold is a click, not a move', () => {
    const th = /** @type {Element} */ (region.querySelector('th[data-col="section"]'));
    fire('pointerdown', th, 150);
    fire('pointermove', th, 152);
    fire('pointerup', th, 152);
    expect(calls.move).toEqual([]);
    expect(document.querySelector('.col-drag-ghost')).toBeNull();
  });

  test('pinned headers and their controls start no gesture', () => {
    const box = /** @type {Element} */ (region.querySelector('th[data-col="chk"] input'));
    fire('pointerdown', box, 5);
    fire('pointermove', box, 250);
    fire('pointerup', box, 250);
    expect(calls.move).toEqual([]);
  });

  test('double-clicking an edge asks for an autosize', () => {
    const grip = /** @type {Element} */ (region.querySelector('[data-resize="address"]'));
    grip.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true }));
    expect(calls.autosize).toEqual(['address']);
  });

  test('detach removes every listener', () => {
    detach();
    const grip = /** @type {Element} */ (region.querySelector('[data-resize="section"]'));
    fire('pointerdown', grip, 200);
    fire('pointermove', grip, 260);
    fire('pointerup', grip, 260);
    expect(calls.resize).toEqual([]);
  });
});

describe('measureColumn', () => {
  test('fits the widest cell of the column, header included, skipping message rows', () => {
    document.body.innerHTML = `<table>
      <thead><tr><th data-col="a">A</th><th data-col="b">B</th></tr></thead>
      <tbody><tr><td>x</td><td>wide</td></tr><tr><td colspan="2">No channels match.</td></tr></tbody></table>`;
    const table = /** @type {HTMLTableElement} */ (document.querySelector('table'));
    const widths = new Map([
      ['A', 30],
      ['B', 20],
      ['x', 10],
      ['wide', 120],
      ['No channels match.', 900],
    ]);
    for (const cell of table.querySelectorAll('th, td')) {
      Object.defineProperty(cell, 'scrollWidth', {
        value: widths.get(cell.textContent ?? '') ?? 0,
      });
    }
    expect(measureColumn(table, 'a')).toBe(32);
    expect(measureColumn(table, 'b')).toBe(122);
    expect(measureColumn(table, 'missing')).toBeNull();
  });
});
