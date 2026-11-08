import { beforeEach, describe, expect, it } from 'vitest';
import { panelOf, showState } from '../panel/panel.js';

// Zoom on the strip: wheel around the cursor, drag-select, pan when zoomed,
// shift-drag selects, double-click or the "whole ring" link resets, the header
// chip names the visible range, and the zoomed view labels every member.

/**
 * @typedef {import('../panel/panel.js').Member} Member
 * @typedef {import('../panel/panel.js').Model} Model
 * @typedef {import('../panel/panel.js').BumpState} BumpState
 */

const MARKUP = `
  <span data-field="plane"><button data-plane="x"></button><button data-plane="y"></button></span>
  <select data-field="model" hidden></select>
  <span data-field="orm"><i></i><span data-field="orm-text"></span><button data-field="recompute"></button></span>
  <span data-field="no-model" hidden>no closed-orbit model</span>
  <p data-field="posture" hidden></p>
  <p data-field="notice" hidden></p>
  <p data-field="error" hidden></p>
  <div data-field="body"></div>
  <div data-field="confirm" hidden></div>`;

const LENGTH = 100;
// Strip geometry in SVG user units, as panel.js draws it.
const LEFT = 34;
const PLOT_W = 1000 - 34 - 12;

/**
 * @param {string} element
 * @param {'x' | 'y'} plane
 * @param {number} index
 * @param {number} s
 * @returns {Member}
 */
function bpm(element, plane, index, s) {
  return {
    element,
    plane,
    index,
    s_m: s,
    unit: 'mm',
    position_addr: `${element}:POS:${plane}`,
    golden_addr: `${element}:GOLD:${plane}`,
  };
}

/**
 * @param {string} element
 * @param {'x' | 'y'} plane
 * @param {number} index
 * @param {number} s
 * @returns {Member}
 */
function corrector(element, plane, index, s) {
  return {
    element,
    plane,
    index,
    s_m: s,
    unit: 'A',
    setpoint_addr: `${element}:SP`,
    readback_addr: `${element}:RB`,
  };
}

const BPMS = [
  bpm('B1', 'x', 1, 10),
  bpm('B2', 'x', 4, 30),
  bpm('B3', 'x', 7, 50),
  bpm('B4', 'x', 9, 80),
];

const CORRECTORS = [
  corrector('H1', 'x', 2, 20),
  corrector('H2', 'x', 5, 40),
  corrector('H3', 'x', 8, 70),
];

/**
 * @param {number} [length]
 * @returns {Model}
 */
function model(length = LENGTH) {
  return {
    name: 'RING',
    length_m: length,
    orm: { status: 'ready', age_s: 3, version: 1, error: null, unsolved: [] },
    members: { bpms: BPMS, correctors: CORRECTORS },
  };
}

/** @returns {{position: Record<string, number | null>, golden: Record<string, number | null>}} */
function orbit() {
  /** @type {Record<string, number | null>} */
  const position = {};
  /** @type {Record<string, number | null>} */
  const golden = {};
  for (const member of BPMS) {
    if (member.position_addr) position[member.position_addr] = 0.2;
    if (member.golden_addr) golden[member.golden_addr] = 0.1;
  }
  return { position, golden };
}

/**
 * @param {Partial<BumpState>} [over]
 * @returns {BumpState}
 */
function state(over = {}) {
  return {
    models: [model()],
    posture: { allowed: true, reason: null, phrase: '' },
    orbit: orbit(),
    plan: null,
    ...over,
  };
}

/** @param {Element} root */
function svg(root) {
  const strip = root.querySelector('[data-field="body"] svg[data-field="strip"]');
  if (!strip) throw new Error('no strip');
  return strip;
}

/** @param {Element} root */
function background(root) {
  const found = svg(root).querySelector('.ob-bg');
  if (!found) throw new Error('no background');
  return found;
}

/**
 * Give the strip a layout box (happy-dom does none): `left` and `width` in
 * client pixels.
 * @param {Element} root
 * @param {number} [width]
 * @param {number} [left]
 */
function layout(root, width = 1000, left = 0) {
  const box = {
    left,
    top: 0,
    width,
    height: width / 4,
    right: left + width,
    bottom: width / 4,
    x: left,
    y: 0,
    toJSON() {
      return {};
    },
  };
  svg(root).getBoundingClientRect = () => /** @type {DOMRect} */ (box);
}

/**
 * The client x of an s-position on the strip for the given view, with the
 * default 1000-px box at left 0 (client px == SVG user units).
 * @param {number} s
 * @param {number} [s0]
 * @param {number} [s1]
 */
function xAt(s, s0 = 0, s1 = LENGTH) {
  return LEFT + ((s - s0) * PLOT_W) / (s1 - s0);
}

/**
 * @param {Element} target
 * @param {number} clientX
 * @param {number} deltaY
 */
function wheel(target, clientX, deltaY) {
  const event = new WheelEvent('wheel', { deltaY, clientX, bubbles: true, cancelable: true });
  // happy-dom's WheelEvent does not extend MouseEvent as browsers' does: no clientX.
  if (event.clientX == null) Object.defineProperty(event, 'clientX', { value: clientX });
  target.dispatchEvent(event);
  return event;
}

/**
 * @param {Element} target
 * @param {string} type
 * @param {number} clientX
 * @param {boolean} [shiftKey]
 */
function pointer(target, type, clientX, shiftKey = false) {
  target.dispatchEvent(
    new PointerEvent(type, { clientX, clientY: 50, pointerId: 1, bubbles: true, shiftKey })
  );
}

/**
 * A drag from x0 to x1 that starts on the empty plot.
 * @param {Element} root
 * @param {number} x0
 * @param {number} x1
 * @param {boolean} [shiftKey]
 */
function drag(root, x0, x1, shiftKey = false) {
  pointer(background(root), 'pointerdown', x0, shiftKey);
  pointer(svg(root), 'pointermove', (x0 + x1) / 2, shiftKey);
  pointer(svg(root), 'pointermove', x1, shiftKey);
  pointer(svg(root), 'pointerup', x1, shiftKey);
}

/** @param {Element} root */
function view(root) {
  return panelOf(root).view;
}

/** @param {Element} root */
function chipText(root) {
  return root.querySelector('[data-field="range"]')?.textContent ?? null;
}

/** @param {Element} root */
function resetLink(root) {
  const link = root.querySelector('[data-field="whole-ring"]');
  if (!(link instanceof HTMLElement)) throw new Error('no whole-ring link');
  return link;
}

/**
 * Names of the labels drawn on the strip.
 * @param {Element} root
 */
function labels(root) {
  return [...svg(root).querySelectorAll('.ob-label')].map((l) => l.getAttribute('data-label'));
}

/**
 * Names of the BPM dots and corrector ticks drawn on the strip.
 * @param {Element} root
 */
function members(root) {
  return [...svg(root).querySelectorAll('.ob-bpm, .ob-cor')].map((m) =>
    m.getAttribute('data-element')
  );
}

/**
 * @param {{s0: number, s1: number} | null} actual
 * @param {number} s0
 * @param {number} s1
 */
function expectView(actual, s0, s1) {
  expect(actual).not.toBeNull();
  expect(actual?.s0).toBeCloseTo(s0, 6);
  expect(actual?.s1).toBeCloseTo(s1, 6);
}

describe('zoom on the orbit-bump strip', () => {
  /** @type {HTMLElement} */
  let root;

  beforeEach(() => {
    root = document.createElement('main');
    root.innerHTML = MARKUP;
    showState(root, state());
    layout(root);
  });

  it('opens on the whole ring: chip says so, no reset link, no labels', () => {
    expect(view(root)).toBeNull();
    expect(chipText(root)).toBe('whole ring');
    expect(resetLink(root).hidden).toBe(true);
    expect(labels(root)).toEqual([]);
  });

  it('zooms in on the wheel around the cursor', () => {
    const event = wheel(svg(root), xAt(50), -100);
    expect(event.defaultPrevented).toBe(true);
    // 0.8 × 100 m around s = 50, which stays under the cursor.
    expectView(view(root), 10, 90);
    expect(Number(svg(root).getAttribute('data-s0'))).toBeCloseTo(10, 6);
    expect(Number(svg(root).getAttribute('data-s1'))).toBeCloseTo(90, 6);
  });

  it('keeps the s under the cursor fixed off-centre too', () => {
    wheel(svg(root), xAt(20), -100);
    const zoomed = view(root);
    expectView(zoomed, 4, 84);
  });

  it('maps the cursor through the laid-out box, not the viewBox', () => {
    // A 500-px box starting at client x 100: user unit u sits at 100 + u / 2.
    layout(root, 500, 100);
    wheel(svg(root), 100 + xAt(50) / 2, -100);
    expectView(view(root), 10, 90);
  });

  it('zooms out on the wheel and returns to the whole ring at full span', () => {
    wheel(svg(root), xAt(50), -100);
    wheel(svg(root), xAt(50, 10, 90), -100);
    expectView(view(root), 18, 82);
    wheel(svg(root), xAt(50, 18, 82), 100);
    expectView(view(root), 10, 90);
    wheel(svg(root), xAt(50, 10, 90), 100);
    expect(view(root)).toBeNull();
    expect(chipText(root)).toBe('whole ring');
  });

  it('clamps a wheel zoom near the edge inside the ring', () => {
    wheel(svg(root), xAt(50), -100);
    // Zooming out with the cursor at the right edge would push s1 past the ring.
    wheel(svg(root), xAt(90, 10, 90), 100);
    expect(view(root)).toBeNull();
    wheel(svg(root), xAt(98), -100);
    const zoomed = view(root);
    expect(zoomed?.s0 ?? -1).toBeGreaterThanOrEqual(0);
    expect(zoomed?.s1 ?? 1e9).toBeLessThanOrEqual(LENGTH);
  });

  it('zooms to a range dragged across the empty plot', () => {
    pointer(background(root), 'pointerdown', xAt(20));
    pointer(svg(root), 'pointermove', xAt(60));
    // The selection is drawn while the drag is in flight.
    expect(svg(root).querySelector('.ob-zoom-select')).not.toBeNull();
    expect(view(root)).toBeNull();
    pointer(svg(root), 'pointerup', xAt(60));
    expectView(view(root), 20, 60);
    expect(svg(root).querySelector('.ob-zoom-select')).toBeNull();
  });

  it('zooms to a range dragged right to left as well', () => {
    drag(root, xAt(60), xAt(20));
    expectView(view(root), 20, 60);
  });

  it('ignores a drag too short to be a range', () => {
    drag(root, xAt(40), xAt(40) + 2);
    expect(view(root)).toBeNull();
  });

  it('names the visible range in the header chip and offers the reset link', () => {
    drag(root, xAt(25), xAt(60));
    const text = chipText(root) ?? '';
    expect(text).toContain('B2–B3');
    expect(text).toContain('25');
    expect(text).toContain('60');
    expect(text).not.toBe('whole ring');
    expect(resetLink(root).hidden).toBe(false);
  });

  it('labels every BPM and corrector in range when zoomed', () => {
    drag(root, xAt(25), xAt(60));
    const shown = members(root);
    expect(shown.sort()).toEqual(['B2', 'B3', 'H2']);
    expect(labels(root).sort()).toEqual(shown.sort());
  });

  it('pans when zoomed and the drag starts on the empty plot', () => {
    drag(root, xAt(20), xAt(60));
    // Dragging right by 10 m worth of strip moves the window 10 m left.
    const x0 = xAt(40, 20, 60);
    drag(root, x0, x0 + (10 * PLOT_W) / 40);
    expectView(view(root), 10, 50);
    expect(svg(root).querySelector('.ob-zoom-select')).toBeNull();
  });

  it('stops a pan at the ring edge, keeping the span', () => {
    drag(root, xAt(20), xAt(60));
    const x0 = xAt(40, 20, 60);
    drag(root, x0, x0 + (50 * PLOT_W) / 40);
    expectView(view(root), 0, 40);
  });

  it('selects again on a shift-drag when zoomed', () => {
    drag(root, xAt(20), xAt(60));
    drag(root, xAt(30, 20, 60), xAt(50, 20, 60), true);
    expectView(view(root), 30, 50);
  });

  it('starts no drag from a BPM dot', () => {
    const dot = svg(root).querySelector('.ob-bpm[data-element="B2"]');
    if (!dot) throw new Error('no dot');
    pointer(dot, 'pointerdown', xAt(30));
    pointer(svg(root), 'pointermove', xAt(70));
    pointer(svg(root), 'pointerup', xAt(70));
    expect(view(root)).toBeNull();
    expect(svg(root).querySelector('.ob-zoom-select')).toBeNull();
  });

  it('returns to the whole ring on a double-click', () => {
    drag(root, xAt(25), xAt(60));
    svg(root).dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(view(root)).toBeNull();
    expect(svg(root).getAttribute('data-s0')).toBe('0');
    expect(svg(root).getAttribute('data-s1')).toBe(String(LENGTH));
    expect(chipText(root)).toBe('whole ring');
    expect(resetLink(root).hidden).toBe(true);
    expect(labels(root)).toEqual([]);
  });

  it('returns to the whole ring from the chip link', () => {
    drag(root, xAt(25), xAt(60));
    resetLink(root).dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(view(root)).toBeNull();
    expect(chipText(root)).toBe('whole ring');
  });

  it('keeps the view through a poll', () => {
    drag(root, xAt(25), xAt(60));
    showState(root, state());
    expectView(view(root), 25, 60);
    expect(labels(root).length).toBeGreaterThan(0);
  });

  it('leaves the view unchanged when the strip has no width', () => {
    layout(root, 0);
    wheel(svg(root), 500, -100);
    expect(view(root)).toBeNull();
    drag(root, 200, 600);
    expect(view(root)).toBeNull();

    layout(root);
    drag(root, xAt(20), xAt(60));
    layout(root, 0);
    wheel(svg(root), 500, -100);
    drag(root, 200, 600);
    expectView(view(root), 20, 60);
  });

  it('pulls a view back inside a shorter ring', () => {
    drag(root, xAt(60), xAt(95));
    showState(root, state({ models: [model(70)] }));
    const kept = view(root);
    expect(kept?.s1 ?? 1e9).toBeLessThanOrEqual(70);
    expect(kept?.s0 ?? -1).toBeGreaterThanOrEqual(0);
  });
});
