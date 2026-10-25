/**
 * Unit tests for <osprey-machine-map>
 * (design_system/static/js/components/osprey-machine-map.js).
 *
 *   npx vitest run tests/interfaces/design_system/js/machine-map.test.mjs
 *
 * Covers:
 *   - the geometry: a device at position s sits at angle -pi/2 + 2*pi*s/length
 *   - the per-region counts (in use / total)
 *   - the render: one dot per device at its position, the in / out / pending
 *     states, inner lane for a second plane, region arcs and region rows, the
 *     focused region highlighted, the centre text, labels escaped
 *   - the events: a dot click asks for a toggle, a region arc or row click
 *     (or Enter on a row) moves the focus
 *   - nothing names a facility: every label comes from the model
 */

import { test, expect, describe, afterEach } from 'vitest';

import {
  mapPoint,
  regionSummary,
} from '/design-system/js/components/osprey-machine-map.js';

/**
 * A two-region machine of length 100 with three devices.
 * @param {object} [overrides]
 * @returns {any}
 */
function model(overrides = {}) {
  return {
    label: 'Device map',
    length: 100,
    regions: [
      { id: 1, label: 'Region 1', mapLabel: '1', start: 0, end: 50 },
      { id: 2, label: 'Region 2', mapLabel: '2', start: 50, end: 100 },
    ],
    devices: [
      { id: 'a', label: 'Dev A', position: 10, region: 1, state: 'in' },
      { id: 'b', label: 'Dev B', position: 30, region: 1, state: 'out' },
      { id: 'c', label: 'Dev C', position: 75, region: 2, state: 'in', pending: true },
    ],
    focus: 1,
    center: { value: '2 / 3', caption: 'in use' },
    ...overrides,
  };
}

/** @returns {any} */
function mount(m = model()) {
  const el = /** @type {any} */ (document.createElement('osprey-machine-map'));
  el.model = m;
  document.body.appendChild(el);
  return el;
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('geometry', () => {
  test('position 0 sits at the top, a quarter turn to the right', () => {
    const [x0, y0] = mapPoint(0, 100, 10, [50, 50]);
    expect(x0).toBeCloseTo(50);
    expect(y0).toBeCloseTo(40);
    const [x1, y1] = mapPoint(25, 100, 10, [50, 50]);
    expect(x1).toBeCloseTo(60);
    expect(y1).toBeCloseTo(50);
  });

  test('a full length comes back to the start', () => {
    const a = mapPoint(0, 80, 12, [0, 0]);
    const b = mapPoint(80, 80, 12, [0, 0]);
    expect(b[0]).toBeCloseTo(a[0]);
    expect(b[1]).toBeCloseTo(a[1]);
  });

  test('region summary counts devices in use', () => {
    expect(regionSummary(model().devices, 1)).toEqual({ used: 1, total: 2 });
    expect(regionSummary(model().devices, 2)).toEqual({ used: 1, total: 1 });
    expect(regionSummary(model().devices, 3)).toEqual({ used: 0, total: 0 });
  });
});

describe('<osprey-machine-map> render', () => {
  test('registers the element', () => {
    expect(customElements.get('osprey-machine-map')).toBeDefined();
  });

  test('draws one dot per device with its state and pending flag', () => {
    const el = mount();
    const dots = el.querySelectorAll('svg.machine-map .dev');
    expect(dots.length).toBe(3);
    const byId = Object.fromEntries([...dots].map((d) => [d.getAttribute('data-device'), d]));
    expect(byId.a.getAttribute('data-state')).toBe('in');
    expect(byId.b.getAttribute('data-state')).toBe('out');
    expect(byId.c.getAttribute('data-pending')).toBe('true');
    expect(byId.a.getAttribute('data-pending')).toBe('false');
    expect(byId.a.querySelector('title')?.textContent).toContain('Dev A');
  });

  test('a device sits on the map at its position', () => {
    const el = mount();
    const svg = el.querySelector('svg.machine-map');
    const [, , w, h] = svg.getAttribute('viewBox').split(' ').map(Number);
    const dot = el.querySelector('.dev[data-device="a"]');
    const r = Number(el.querySelector('.map-path').getAttribute('r'));
    const [x, y] = mapPoint(10, 100, r, [w / 2, h / 2]);
    expect(Number(dot.getAttribute('cx'))).toBeCloseTo(x, 0);
    expect(Number(dot.getAttribute('cy'))).toBeCloseTo(y, 0);
  });

  test('a second lane draws on an inner track', () => {
    const el = mount(
      model({
        devices: [
          { id: 'h', label: 'H1', position: 0, region: 1, lane: 0, state: 'in' },
          { id: 'v', label: 'V1', position: 0, region: 1, lane: 1, state: 'in' },
        ],
      }),
    );
    expect(el.querySelectorAll('.map-path').length).toBe(2);
    const yh = Number(el.querySelector('.dev[data-device="h"]').getAttribute('cy'));
    const yv = Number(el.querySelector('.dev[data-device="v"]').getAttribute('cy'));
    expect(yv).toBeGreaterThan(yh);
    expect(el.querySelectorAll('.sector-row[data-region="1"] .cells.planes .plane').length).toBe(2);
  });

  test('draws a region arc, a map label and a row per region, the focus marked', () => {
    const el = mount();
    expect(el.querySelectorAll('.sector-arc').length).toBe(2);
    expect(el.querySelector('.sector-arc.is-focus')?.getAttribute('data-region')).toBe('1');
    const rows = el.querySelectorAll('.sector-row');
    expect(rows.length).toBe(2);
    expect(rows[0].classList.contains('is-focus')).toBe(true);
    expect(rows[0].querySelector('.sec')?.textContent).toBe('Region 1');
    expect(rows[0].querySelector('.cnt')?.textContent).toBe('1/2');
    expect(rows[0].querySelectorAll('.cell').length).toBe(2);
    const labels = [...el.querySelectorAll('.sector-label')].map((n) => n.textContent);
    expect(labels).toEqual(['1', '2']);
  });

  test('draws the centre text and the marker from the model', () => {
    const el = mount(model({ marker: { position: 1, label: 'Inj' } }));
    expect(el.querySelector('.center-big')?.textContent).toBe('2 / 3');
    expect(el.querySelector('.center-small')?.textContent).toBe('in use');
    expect(el.querySelector('.inj')).not.toBeNull();
    expect(el.querySelector('.inj-label')?.textContent).toBe('Inj');
  });

  test('escapes labels', () => {
    const el = mount(
      model({
        regions: [{ id: 1, label: '<b>x</b>', start: 0, end: 100 }],
        devices: [{ id: 'q', label: '<i>q</i>', position: 1, region: 1, state: 'in' }],
      }),
    );
    expect(el.querySelector('b')).toBeNull();
    expect(el.querySelector('i')).toBeNull();
    expect(el.querySelector('.sector-row .sec')?.textContent).toBe('<b>x</b>');
  });

  test('an empty element renders nothing until it has a model', () => {
    const el = document.createElement('osprey-machine-map');
    document.body.appendChild(el);
    expect(el.querySelector('svg')).toBeNull();
  });
});

describe('<osprey-machine-map> events', () => {
  test('a dot click asks to toggle that device', () => {
    const el = mount();
    /** @type {any[]} */
    const seen = [];
    el.addEventListener('machine-map:toggle', (/** @type {any} */ e) => seen.push(e.detail));
    el.querySelector('.dev[data-device="b"]').dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(seen).toEqual([{ device: 'b' }]);
  });

  test('a region arc click moves the focus and announces it', () => {
    const el = mount();
    /** @type {any[]} */
    const seen = [];
    el.addEventListener('machine-map:focus', (/** @type {any} */ e) => seen.push(e.detail));
    el.querySelector('.sector-arc[data-region="2"]').dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(seen).toEqual([{ region: 2 }]);
    expect(el.model.focus).toBe(2);
    expect(el.querySelector('.sector-row.is-focus')?.getAttribute('data-region')).toBe('2');
  });

  test('Enter on a region row moves the focus', () => {
    const el = mount();
    /** @type {any[]} */
    const seen = [];
    el.addEventListener('machine-map:focus', (/** @type {any} */ e) => seen.push(e.detail));
    const row = el.querySelector('.sector-row[data-region="2"]');
    expect(row.getAttribute('role')).toBe('button');
    expect(row.getAttribute('tabindex')).toBe('0');
    row.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(seen).toEqual([{ region: 2 }]);
  });
});
